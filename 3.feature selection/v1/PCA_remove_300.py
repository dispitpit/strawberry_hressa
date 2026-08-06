# 前 3 個主成分解釋的變異比例: [0.85717225 0.0682984  0.04275447]
# 總解釋比例: 96.82251 %
# === 前 10 個重要波段（PCA1 權重排序）===
# 波段索引: 293, 實際波長: 1001.59 nm, 權重: +0.093538
# 波段索引: 291, 實際波長: 997.53 nm, 權重: +0.091769
# 波段索引: 288, 實際波長: 991.45 nm, 權重: +0.090677
# 波段索引: 289, 實際波長: 993.48 nm, 權重: +0.090581
# 波段索引: 283, 實際波長: 981.33 nm, 權重: +0.089931
# 波段索引: 286, 實際波長: 987.40 nm, 權重: +0.089306
# 波段索引: 287, 實際波長: 989.43 nm, 權重: +0.088809
# 波段索引: 290, 實際波長: 995.50 nm, 權重: +0.088183
# 波段索引: 217, 實際波長: 848.63 nm, 權重: +0.087570
# 波段索引: 216, 實際波長: 846.63 nm, 權重: +0.087486
# 圖片已儲存至 PCA_Projection_remove_300.png
# PCA 投影結果已儲存為 pca_projection_results_remove_300.csv

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
healthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_SPA/healthy")
unhealthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_SPA/unhealthy")

X = []
y = []
file_names = []

for f in healthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Healthy")
    file_names.append(f.name)

for f in unhealthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Unhealthy")
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# === PCA 解釋比例 ===
explained_variance = pca.explained_variance_ratio_
print("前 3 個主成分解釋的變異比例:", explained_variance)
print("總解釋比例:", np.sum(explained_variance) * 100, "%")

# === 找出 PCA1 的重要波段 ===
components = pca.components_[0]
abs_components = np.abs(components)
important_indices = abs_components.argsort()[::-1][:10]

# 讀取波長
wavelengths = np.load("wavelengths_remove_bands_300.npy")

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

save_path = f"PCA_Projection_remove_300.png"
plt.savefig(save_path, dpi=300)
print(f"圖片已儲存至 {save_path}")
plt.show()

pca_df = pd.DataFrame({
    "File": file_names,
    "Label": y,
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "PCA3": X_pca[:, 2]
})
pca_df.to_csv("pca_projection_results_remove_300.csv", index=False)
print("PCA 投影結果已儲存為 pca_projection_results_remove_300.csv")

print("finish")
