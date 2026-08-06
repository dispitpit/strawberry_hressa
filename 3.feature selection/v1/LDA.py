# === 前 20 個重要波段（LDA 權重排序）===
# 波段索引: 136, 實際波長: 673.59 nm, 權重: -1011426.3750
# 波段索引: 127, 實際波長: 655.29 nm, 權重: 936027.5625
# 波段索引: 120, 實際波長: 641.02 nm, 權重: 817110.9375
# 波段索引: 123, 實際波長: 647.14 nm, 權重: 809750.6250
# 波段索引: 109, 實際波長: 618.52 nm, 權重: 803471.1250
# 波段索引: 106, 實際波長: 612.37 nm, 權重: 768259.8750
# 波段索引: 85, 實際波長: 569.07 nm, 權重: 738628.9375
# 波段索引: 116, 實際波長: 632.85 nm, 權重: -717997.5625
# 波段索引: 103, 實際波長: 606.21 nm, 權重: -710474.3125
# 波段索引: 131, 實際波長: 663.43 nm, 權重: -686475.3750
# 波段索引: 117, 實際波長: 634.90 nm, 權重: 667872.1875
# 波段索引: 143, 實際波長: 687.78 nm, 權重: 665822.1250
# 波段索引: 124, 實際波長: 649.18 nm, 權重: -631896.9375
# 波段索引: 126, 實際波長: 653.26 nm, 權重: -619409.7500
# 波段索引: 121, 實際波長: 643.06 nm, 權重: -599541.6250
# 波段索引: 129, 實際波長: 659.36 nm, 權重: -577276.3750
# 波段索引: 98, 實際波長: 595.92 nm, 權重: 576869.1875
# 波段索引: 135, 實際波長: 671.56 nm, 權重: 570829.5625
# 波段索引: 64, 實際波長: 525.31 nm, 權重: 511292.8125
# 波段索引: 94, 實際波長: 587.68 nm, 權重: 510855.5938
# 已儲存前 20 重要波段為 lda_top_bands.csv

#====================================================
# 正權重（+）：該波段的值越高，越傾向於分類到一個特定類別（例如 Unhealthy）。
# 負權重（−）：該波段的值越高，越傾向於分類到另一個類別（例如 Healthy）。
#====================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# 設定全域字型
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# === 讀取資料 ===
healthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA\npy_LDA\healthy")
unhealthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA\npy_LDA\unhealthy")

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

# === LDA 降維 ===
lda = LDA(n_components=1)  # 可以改成2看 2D 視覺化
X_lda = lda.fit_transform(X, y)

# === 繪圖 ===
plt.figure(figsize=(8, 6))
for label, color in zip(["Healthy", "Unhealthy"], ["blue", "red"]):
    plt.scatter(X_lda[y == label], np.zeros_like(X_lda[y == label]), label=label, alpha=0.7, color=color)

plt.title("LDA Projection (1D)")
plt.xlabel("LDA Component 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
# === 儲存圖檔 ===
save_path = f"LDA_Projection.png"
plt.savefig(save_path, dpi=300)
print(f"圖片已儲存至 {save_path}")
plt.show()

# === LDA 投影結果保存 ===
lda_df = pd.DataFrame({
    "File": file_names,
    "Label": y,
    "LDA1": X_lda.flatten()
})
lda_df.to_csv("lda_projection_results.csv", index=False)
print("LDA 投影結果已儲存為 lda_projection_results.csv")

# === LDA 係數（每個波段的權重）===
lda_weights = lda.coef_[0]  # shape: (波段數,)

# === 對應波段名稱 ===
# wavelengths.npy 檔案儲存實際波段資訊
wavelengths = np.load("wavelengths.npy")  # 這是 shape=(300,) 的 array

# === 找出前 10 個最重要波段 ===
top_band_indices = np.argsort(np.abs(lda_weights))[::-1][:60]

print("\n=== 前 10 個重要波段（LDA 權重排序）===")
for idx in top_band_indices:
    print(f"波段索引: {idx}, 實際波長: {wavelengths[idx]:.2f} nm, 權重: {lda_weights[idx]:.4f}")

# === 儲存重要波段資訊 ===
top_bands_info = pd.DataFrame({
    "Band Index": top_band_indices,
    "Wavelength (nm)": wavelengths[top_band_indices],
    "LDA Weight": lda_weights[top_band_indices]
})
top_bands_info.to_csv("lda_top_bands.csv", index=False)
print("已儲存前 10 重要波段為 lda_top_bands.csv")
