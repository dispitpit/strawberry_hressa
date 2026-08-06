# 選定波段索引: [125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
# 對應波長: [651.220239 653.257287 655.293621 657.329251 659.364184 661.398426
#  663.431987 665.464874 667.497094 669.528656 671.559566 673.589833
#  675.619464 677.648468 679.676851 681.704621 683.731787 685.758355
#  687.784334 689.809731 691.834554 693.858811 695.882509 697.905656
#  699.92826 ]
# LDA 圖片已儲存至 LDA_Projection_650_700nm.png
# LDA 投影結果已儲存為 lda_projection_650_700nm.csv

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# 設定全域字型
plt.rcParams["font.family"] = "Times New Roman"


# === 1. 讀取 wavelengths ===
wavelengths = np.load(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/wavelengths.npy")

# 找出 700~750nm 對應波段索引
selected_band_indices = [i for i, w in enumerate(wavelengths) if 650 <= w <= 700]
print(f"選定波段索引: {selected_band_indices}")
print(f"對應波長: {wavelengths[selected_band_indices]}")

# === 2. 讀取資料 ===
healthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_LDA/healthy")
unhealthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_LDA/unhealthy")

X = []
y = []
file_names = []

# Healthy
for f in healthy_dir.glob("*.npy"):
    spec = np.load(f)
    X.append(spec[selected_band_indices])
    y.append("Healthy")
    file_names.append(f.name)

# Unhealthy
for f in unhealthy_dir.glob("*.npy"):
    spec = np.load(f)
    X.append(spec[selected_band_indices])
    y.append("Unhealthy")
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)

lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(8, 6))
for label, color in zip(["Healthy", "Unhealthy"], ["blue", "red"]):
    plt.scatter(X_lda[y == label], np.zeros_like(X_lda[y == label]), label=label, alpha=0.7, color=color)

plt.title("LDA Projection (650-700nm)", fontsize=14)
plt.xlabel("LDA Component 1", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = "LDA_Projection_650_700nm.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"LDA 圖片已儲存至 {save_path}")

lda_df = pd.DataFrame({
    "File": file_names,
    "Label": y,
    "LDA1": X_lda.flatten()
})
lda_df.to_csv("lda_projection_650_700nm.csv", index=False)
print("LDA 投影結果已儲存為 lda_projection_650_700nm.csv")
