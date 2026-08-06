'''
mask = (wavelengths >= 450.0)  & (wavelengths <= 950.0)
原始 shape: (225, 300) → 篩選後 shape: (225, 245)
前 3 個主成分解釋的變異比例: [0.9043902  0.05829868 0.02215318]
總解釋比例: 98.484215 %
=== 前 10 個重要波段（PCA1 權重排序，≥400 nm）===
波段索引(篩後): 194, 實際波長: 848.63 nm, 權重: +0.098866
波段索引(篩後): 193, 實際波長: 846.63 nm, 權重: +0.098799
波段索引(篩後): 161, 實際波長: 782.48 nm, 權重: +0.098735
波段索引(篩後): 162, 實際波長: 784.49 nm, 權重: +0.098674
波段索引(篩後): 202, 實際波長: 864.67 nm, 權重: +0.098564
波段索引(篩後): 196, 實際波長: 852.64 nm, 權重: +0.098516
波段索引(篩後): 199, 實際波長: 858.65 nm, 權重: +0.098491
波段索引(篩後): 160, 實際波長: 780.48 nm, 權重: +0.098465
波段索引(篩後): 198, 實際波長: 856.65 nm, 權重: +0.098299
波段索引(篩後): 200, 實際波長: 860.66 nm, 權重: +0.098271
圖片已儲存至 PCA_Projection.png
PCA 投影結果已儲存為 pca_projection_results.csv
'''
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from spectral import *

# 設定全域字型
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# === 讀取資料 ===
healthy_dir = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/重製1003/ks_npy_train_subset/healthy")
unhealthy_dir = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0415_LDA/重製1003/ks_npy_train_subset/unhealthy")

X = []
y = []
file_names = []

# 載入 Healthy 樣本
for f in healthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Healthy")
    file_names.append(f.name)

# 載入 Unhealthy 樣本
for f in unhealthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Unhealthy")
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)


wavelengths = np.load("wavelengths.npy")
# --- 只取 ≥400 nm 的波段 ---
mask = (wavelengths >= 450.0)  & (wavelengths <= 950.0)  # True/False 長度 = n_bands
assert wavelengths.ndim == 1
assert X.shape[1] == wavelengths.shape[0], "波長數量需等於光譜欄數"
assert mask.dtype == bool and mask.shape[0] == X.shape[1]

wavelengths_sel = wavelengths[mask]                # 篩過後的波長
X_sel = X[:, mask]                                 # 篩欄：把光譜只留 ≥400nm 的列向特徵

print("原始 shape:", X.shape, "→ 篩選後 shape:", X_sel.shape)

# ===（可選）標準化再做 PCA：讓各波段同尺度 ===
# from sklearn.preprocessing import StandardScaler
# X_sel = StandardScaler().fit_transform(X_sel)

# === PCA（用篩過後的 X_sel）===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_sel)

# 解釋比例
explained_variance = pca.explained_variance_ratio_
print("前 3 個主成分解釋的變異比例:", explained_variance)
print("總解釋比例:", np.sum(explained_variance) * 100, "%")

# === PCA1 的重要波段（記得對應 wavelengths_sel）===
components = pca.components_[0]         # 長度 = X_sel 的特徵數
abs_components = np.abs(components)
important_indices = abs_components.argsort()[::-1][:10]

print("=== 前 10 個重要波段（PCA1 權重排序，≥400 nm）===")
for idx in important_indices:
    print(f"波段索引(篩後): {idx}, 實際波長: {wavelengths_sel[idx]:.2f} nm, 權重: {components[idx]:+.6f}")


# === 繪圖 PCA 投影 ===
plt.figure(figsize=(8, 6))
for label, color in zip(["Healthy", "Unhealthy"], ["blue", "red"]):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=label, alpha=0.7, color=color)

plt.title("PCA Projection (2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
# === 儲存圖檔 ===
save_path = f"PCA_Projection.png"
plt.savefig(save_path, dpi=300)
print(f"圖片已儲存至 {save_path}")
plt.show()

# === PCA 投影結果保存 ===
pca_df = pd.DataFrame({
    "File": file_names,
    "Label": y,
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "PCA3": X_pca[:, 2]
})
pca_df.to_csv("pca_projection_results.csv", index=False)
print("PCA 投影結果已儲存為 pca_projection_results.csv")
