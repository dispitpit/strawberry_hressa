
# RF
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 設定字型
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

# --- 讀波長 + 建 mask（例：450–950 nm）---
wavelengths = np.load("wavelengths.npy").astype(float)   # shape=(n_bands,)
mask = (wavelengths >= 450.0) & (wavelengths <= 950.0)

# 安全檢查 + 套遮罩
assert X.shape[1] == wavelengths.shape[0], "波長數量需等於光譜欄數"
X_sel = X[:, mask]
wavelengths_sel = wavelengths[mask]
orig_cols = np.flatnonzero(mask)  # 篩後→原始索引對照
print("原始:", X.shape, "→ 篩後:", X_sel.shape, "保留波段數:", mask.sum())

# === 訓練 Random Forest ===
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X, y)

rf = RandomForestClassifier(
    n_estimators=700,       # 森林裡有 300 棵樹
    max_depth=10,           # 每棵樹的最大深度為 10
    min_samples_leaf=2,     # 每個葉節點至少有 3 筆樣本
    max_features="sqrt",    # 每次分裂時考慮 sqrt(特徵數) 個波段
    oob_score=True,         # 開啟袋外評估
    random_state=0,        # 固定隨機種子
    n_jobs=-1               # 用多核加速
)
rf.fit(X_sel, y)
print("OOB Score:", rf.oob_score_)


# --- 特徵重要性（對應篩後特徵）---
importances = rf.feature_importances_
top_k = 20
top_idx_sel = np.argsort(importances)[::-1][:top_k]

print(f"\n=== 前 {top_k} 個重要波段（RF, 450–950 nm）===")
for i in top_idx_sel:
    print(f"篩後索引: {i:3d} | 原始索引: {orig_cols[i]:3d} | "
          f"λ={wavelengths_sel[i]:.2f} nm | 重要性={importances[i]:.6f}")

# 存表（含原始索引，方便回溯）
pd.DataFrame({
    "Band Index (after mask)": top_idx_sel,
    "Band Index (original)": orig_cols[top_idx_sel],
    "Wavelength (nm)": wavelengths_sel[top_idx_sel],
    "RF Importance": importances[top_idx_sel]
}).to_csv("rf_top_bands.csv", index=False)

# --- 繪圖：用波長做 X 軸更直觀 ---
plt.figure(figsize=(10,6))
plt.bar(wavelengths_sel, importances, width= (wavelengths_sel[1]-wavelengths_sel[0]) if len(wavelengths_sel)>1 else 1)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance (masked 450–950 nm)")
plt.grid(True); plt.tight_layout()
plt.savefig("RF_Feature_Importance.png", dpi=300); plt.show()
'''
篩後索引:   2 | 原始索引:  31 | λ=455.45 nm | 重要性=0.014381
篩後索引:   6 | 原始索引:  35 | λ=464.00 nm | 重要性=0.013551
篩後索引:   5 | 原始索引:  34 | λ=461.87 nm | 重要性=0.012661
篩後索引:   4 | 原始索引:  33 | λ=459.73 nm | 重要性=0.011057
篩後索引:   3 | 原始索引:  32 | λ=457.59 nm | 重要性=0.010420
篩後索引:   1 | 原始索引:  30 | λ=453.31 nm | 重要性=0.010148
篩後索引:   9 | 原始索引:  38 | λ=470.39 nm | 重要性=0.009850
篩後索引:   7 | 原始索引:  36 | λ=466.13 nm | 重要性=0.009719
篩後索引:  29 | 原始索引:  58 | λ=512.71 nm | 重要性=0.009171
篩後索引:  13 | 原始索引:  42 | λ=478.90 nm | 重要性=0.009139
篩後索引:  12 | 原始索引:  41 | λ=476.78 nm | 重要性=0.008811
篩後索引: 124 | 原始索引: 153 | λ=708.01 nm | 重要性=0.008733
篩後索引:  18 | 原始索引:  47 | λ=489.50 nm | 重要性=0.008707
篩後索引:  10 | 原始索引:  39 | λ=472.52 nm | 重要性=0.008649
篩後索引:  14 | 原始索引:  43 | λ=481.02 nm | 重要性=0.008476
篩後索引:   8 | 原始索引:  37 | λ=468.26 nm | 重要性=0.007985
篩後索引:  32 | 原始索引:  61 | λ=519.01 nm | 重要性=0.007841
篩後索引:  11 | 原始索引:  40 | λ=474.65 nm | 重要性=0.007737
篩後索引: 244 | 原始索引: 273 | λ=949.03 nm | 重要性=0.007688
篩後索引: 243 | 原始索引: 272 | λ=947.01 nm | 重要性=0.007636
'''