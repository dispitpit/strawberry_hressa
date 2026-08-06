'''
已儲存為 CARS_selected_bands.csv
已儲存圖檔 CARS_Selected_Bands.png
LASSO 選到 8 個波段
(篩後)索引:   1 | (原始)索引:  30 | λ=453.31 nm | LASSO 係數=0.137175
(篩後)索引:  32 | (原始)索引:  61 | λ=519.01 nm | LASSO 係數=-0.485930
(篩後)索引:  58 | (原始)索引:  87 | λ=573.21 nm | LASSO 係數=0.125682
(篩後)索引:  92 | (原始)索引: 121 | λ=643.06 nm | LASSO 係數=0.071064
(篩後)索引: 126 | (原始)索引: 155 | λ=712.05 nm | LASSO 係數=0.168613
(篩後)索引: 151 | (原始)索引: 180 | λ=762.40 nm | LASSO 係數=-0.131764
(篩後)索引: 178 | (原始)索引: 207 | λ=816.57 nm | LASSO 係數=-0.170406
(篩後)索引: 244 | (原始)索引: 273 | λ=949.03 nm | LASSO 係數=0.282967
'''

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# === 讀取資料 ===
healthy_dir = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/重製1003/ks_npy_train_subset/healthy")
unhealthy_dir = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/重製1003/ks_npy_train_subset/unhealthy")

X = []
y = []
file_names = []

for f in healthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append(0)  # 0 = Healthy
    file_names.append(f.name)

for f in unhealthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append(1)  # 1 = Unhealthy
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)

wavelengths = np.load("wavelengths.npy").astype(float)  # (n_bands,)
mask = (wavelengths >= 450.0) & (wavelengths <= 950.0)

assert X.shape[1] == wavelengths.shape[0], "波長數量需等於光譜欄數"
X_m = X[:, mask]                     # 之後 CARS/LASSO 都用這個
wl_m = wavelengths[mask]             # 篩後波長
orig_cols = np.flatnonzero(mask)     # 篩後→原始欄位索引對照
print("原始:", X.shape, "→ 篩後:", X_m.shape, "保留波段數:", mask.sum())

# === CARS 參數設定 ===
n_iterations = 100
pls_components = 2  # 可調整
initial_bands = X_m.shape[1]

# === 初始化 ===
remaining_idx = np.arange(initial_bands)
best_score = -np.inf
best_subset = remaining_idx

# print(f"開始 CARS, 初始 {initial_bands} 個波段")

for iter in range(n_iterations):
    print(f"\n迭代 {iter+1}/{n_iterations}，剩餘 {len(remaining_idx)} 波段")
    X_subset = X_m[:, remaining_idx]

    # 跑 PLS
    pls = PLSRegression(n_components=pls_components)
    pls.fit(X_subset, y)

    # 計算 Cross-Validation
    scores = cross_val_score(pls, X_subset, y, cv=5, scoring='r2')
    avg_score = np.mean(scores)
    print(f"   - CV R^2: {avg_score:.4f}")

    # 更新
    if avg_score > best_score:
        best_score = avg_score
        best_subset = remaining_idx.copy()
        print("[更新最佳波段組合]")

    # PLS 權重排重要性
    coef = np.abs(pls.coef_[:, 0])
    sorted_idx = np.argsort(coef)
    n_remove = max(1, int(0.1 * len(remaining_idx)))  # 每輪刪 10%
    remove_idx = sorted_idx[:n_remove]
    remaining_idx = np.delete(remaining_idx, remove_idx)

    if len(remaining_idx) <= pls_components:
        print("[波段數過少，自動停止]")
        break

# === 結果 ===
print("\n=== CARS 結果 ===")
print(f"最佳 CV R^2: {best_score:.4f}")
print(f"選到的波段索引: {best_subset}")
print(f"選到的實際波長: {wl_m[best_subset]}")

# === 儲存結果 ===
result_df = pd.DataFrame({
    "Band Index (after mask)": best_subset,
    "Band Index (original)": orig_cols[best_subset],
    "Wavelength (nm)": wl_m[best_subset]
})
result_df.to_csv("CARS_selected_bands.csv", index=False)
print("已儲存為 CARS_selected_bands.csv")

# === 畫出選到的波段 ===
plt.figure(figsize=(12, 6))
# plt.plot(wl_m, np.zeros_like(wavelengths), 'k.', alpha=0.3)
# plt.scatter(wavelengths[best_subset], np.zeros_like(best_subset), color='red', label='Selected Bands')

plt.plot(wavelengths, np.zeros_like(wavelengths), 'k.', alpha=0.3)
plt.scatter(wavelengths[orig_cols[best_subset]],
            np.zeros_like(best_subset, dtype=float),
            color='red', label='Selected Bands')


plt.xlabel("Wavelength (nm)")
plt.title("CARS Selected Bands")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("CARS_Selected_Bands.png", dpi=300)
# print("已儲存圖檔 CARS_Selected_Bands.png")
plt.show()


# === CARS 結束 ===
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# === 用 CARS 結果過濾 X ===
X_cars = X_m[:, best_subset]

# 對 X_cars 做標準化（mean=0, std=1）
scaler = StandardScaler()
X_cars_scaled = scaler.fit_transform(X_cars)

# === 跑 LASSO ===
# lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
# lasso.fit(X_cars_scaled, y)

lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_cars_scaled, y)

coef = lasso.coef_
selected_idx_within_cars = np.where(coef != 0)[0]
final_selected_after_mask = best_subset[selected_idx_within_cars]
final_selected_original   = orig_cols[final_selected_after_mask]    # 對回原始索引

print(f"\nLASSO 選到 {len(final_selected_original)} 個波段")
for i_sel, i_org in zip(final_selected_after_mask, final_selected_original):
    print(f"(篩後)索引: {i_sel:3d} | (原始)索引: {i_org:3d} | "
          f"λ={wl_m[i_sel]:.2f} nm | LASSO 係數={coef[selected_idx_within_cars[np.where(final_selected_after_mask==i_sel)[0][0]]]:.6f}")

# === 儲存 ===
pd.DataFrame({
    "Band Index (after mask)": final_selected_after_mask,
    "Band Index (original)": final_selected_original,
    "Wavelength (nm)": wl_m[final_selected_after_mask],
    "LASSO Coef": coef[selected_idx_within_cars]
}).to_csv("LASSO_selected_bands.csv", index=False)

# === 畫圖 ===
plt.figure(figsize=(12, 6))
plt.stem(wl_m[best_subset], coef, use_line_collection=True)
plt.xlabel("Wavelength (nm)")
plt.ylabel("LASSO Coefficient")
plt.title("LASSO Coefficients for CARS-Selected Bands (masked)")
plt.grid(True); plt.tight_layout()
plt.savefig("LASSO_Selected_Bands.png", dpi=300); plt.show()

print("finish")