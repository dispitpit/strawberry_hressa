# 選定波段索引: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124]
# 對應波長: [600.039888 602.097257 604.153721 606.209287 608.263963 610.317757
#  612.370676 614.422729 616.473922 618.524265 620.573763 622.622426
#  624.67026  626.717274 628.763475 630.80887  632.853469 634.897277
#  636.940304 638.982556 641.024042 643.064768 645.104743 647.143975
#  649.182471]
# LDA 圖片已儲存至 LDA_Projection_600_650nm.png
# LDA 投影結果已儲存為 lda_projection_600_650nm.csv
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
selected_band_indices = [i for i, w in enumerate(wavelengths) if 600 <= w <= 650]
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

plt.title("LDA Projection (600-650nm)", fontsize=14)
plt.xlabel("LDA Component 1", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = "LDA_Projection_600_650nm.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"LDA 圖片已儲存至 {save_path}")

lda_df = pd.DataFrame({
    "File": file_names,
    "Label": y,
    "LDA1": X_lda.flatten()
})
lda_df.to_csv("lda_projection_600_650nm.csv", index=False)
print("LDA 投影結果已儲存為 lda_projection_600_650nm.csv")
