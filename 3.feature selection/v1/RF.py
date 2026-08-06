# 前處理: 把400nm以前的波段去除

# === 前 20 個重要波段（RF Feature Importance 排序）===
# 波段索引: 0, 實際波長: 401.50 nm, 重要性: 0.012442
# 波段索引: 1, 實際波長: 403.68 nm, 重要性: 0.011957
# 波段索引: 2, 實際波長: 405.86 nm, 重要性: 0.011418
# 波段索引: 3, 實際波長: 408.03 nm, 重要性: 0.010359
# 波段索引: 13, 實際波長: 429.68 nm, 重要性: 0.009876
# 波段索引: 18, 實際波長: 440.44 nm, 重要性: 0.009472
# 波段索引: 9, 實際波長: 421.04 nm, 重要性: 0.008990
# 波段索引: 29, 實際波長: 464.00 nm, 重要性: 0.008957
# 波段索引: 6, 實際波長: 414.54 nm, 重要性: 0.008947
# 波段索引: 28, 實際波長: 461.87 nm, 重要性: 0.008902
# 波段索引: 17, 實際波長: 438.29 nm, 重要性: 0.008876
# 波段索引: 142, 實際波長: 697.91 nm, 重要性: 0.008721
# 波段索引: 8, 實際波長: 418.87 nm, 重要性: 0.008536
# 波段索引: 27, 實際波長: 459.73 nm, 重要性: 0.008515
# 波段索引: 292, 實際波長: 999.56 nm, 重要性: 0.008359
# 波段索引: 14, 實際波長: 431.84 nm, 重要性: 0.008224
# 波段索引: 59, 實際波長: 527.40 nm, 重要性: 0.007657
# 波段索引: 32, 實際波長: 470.39 nm, 重要性: 0.007548
# 波段索引: 26, 實際波長: 457.59 nm, 重要性: 0.007293
# 波段索引: 24, 實際波長: 453.31 nm, 重要性: 0.007256
# 已儲存前 20 重要波段為 rf_top_bands.csv
# RF 特徵重要性圖已儲存為 RF_Feature_Importance.png


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
healthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_RF/healthy")
unhealthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_RF/unhealthy")

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

# === 訓練 Random Forest ===
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X, y)

rf = RandomForestClassifier(
    n_estimators=500,       # 森林裡有 300 棵樹
    max_depth=10,           # 每棵樹的最大深度為 10
    min_samples_leaf=3,     # 每個葉節點至少有 3 筆樣本
    max_features="sqrt",    # 每次分裂時考慮 sqrt(特徵數) 個波段
    oob_score=True,         # 開啟袋外評估
    random_state=42,        # 固定隨機種子
    n_jobs=-1               # 用多核加速
)
rf.fit(X, y)
print("OOB Score:", rf.oob_score_)

importances = rf.feature_importances_

wavelengths = np.load("wavelengths_remove_bands_300.npy")

top_band_indices = np.argsort(importances)[::-1][:20]

print("\n=== 前 20 個重要波段（RF Feature Importance 排序）===")
for idx in top_band_indices:
    print(f"波段索引: {idx}, 實際波長: {wavelengths[idx]:.2f} nm, 重要性: {importances[idx]:.6f}")

top_bands_info = pd.DataFrame({
    "Band Index": top_band_indices,
    "Wavelength (nm)": wavelengths[top_band_indices],
    "RF Importance": importances[top_band_indices]
})
top_bands_info.to_csv("rf_top_bands.csv", index=False)
print("已儲存前 20 重要波段為 rf_top_bands.csv")

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel("Band Index")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.grid(True)
plt.tight_layout()
plt.savefig("RF_Feature_Importance.png", dpi=300)
print("RF 特徵重要性圖已儲存為 RF_Feature_Importance.png")
plt.show()
