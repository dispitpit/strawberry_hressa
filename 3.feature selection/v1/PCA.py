# 前 3 個主成分解釋的變異比例: [0.6899459  0.25965992 0.02719013]
# 總解釋比例: 97.679596 %
# === 前 10 個重要波段（PCA1 權重排序）===
# 波段索引: 1, 實際波長: 390.60 nm, 權重: +0.270496
# 波段索引: 0, 實際波長: 388.41 nm, 權重: +0.268579
# 波段索引: 2, 實際波長: 392.78 nm, 權重: +0.247672
# 波段索引: 3, 實際波長: 394.96 nm, 權重: +0.220137
# 波段索引: 4, 實際波長: 397.15 nm, 權重: +0.192102
# 波段索引: 5, 實際波長: 399.33 nm, 權重: +0.157164
# 波段索引: 6, 實際波長: 401.50 nm, 權重: +0.123957
# 波段索引: 7, 實際波長: 403.68 nm, 權重: +0.088544
# 波段索引: 299, 實際波長: 1001.59 nm, 權重: -0.084221
# 波段索引: 297, 實際波長: 997.53 nm, 權重: -0.081789
# 圖片已儲存至 PCA_Projection.png
# PCA 投影結果已儲存為 pca_projection_results.csv
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 設定全域字型
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# === 讀取資料 ===
healthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_PCA/healthy")
unhealthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_PCA/unhealthy")

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

# === PCA 降維 ===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# === PCA 解釋比例 ===
explained_variance = pca.explained_variance_ratio_
print("前 3 個主成分解釋的變異比例:", explained_variance)
print("總解釋比例:", np.sum(explained_variance) * 100, "%")

# === 找出 PCA1 的重要波段 ===
components = pca.components_[0]  # 第一主成分的權重
abs_components = np.abs(components)
important_indices = abs_components.argsort()[::-1][:10]  # 取絕對值最大前10個波段

# 讀取波長 (假設從 hdr 檔讀取)
hdr_path = Path(r"C:/Users/Amanda/PycharmProjects/test/病害高光譜2024June/20240620/sample1.hdr")
from spectral import *
wavelengths = open_image(str(hdr_path)).bands.centers

print("=== 前 10 個重要波段（PCA1 權重排序）===")
for idx in important_indices:
    print(f"波段索引: {idx}, 實際波長: {wavelengths[idx]:.2f} nm, 權重: {components[idx]:+.6f}")

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
