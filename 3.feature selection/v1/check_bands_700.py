# 選定波段索引: [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
# 對應波長: [701.950329 703.971869 705.99289  708.013398 710.033401 712.052907
#  714.071924 716.090459 718.108521 720.126116 722.143253 724.159938
#  726.176181 728.191988 730.207368 732.222328 734.236875 736.251018
#  738.264764 740.27812  742.291096 744.303697 746.315932 748.327809]
# LDA 圖片已儲存至 LDA_Projection_700_750nm.png
# LDA 投影結果已儲存為 lda_projection_700_750nm.csv

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
selected_band_indices = [i for i, w in enumerate(wavelengths) if 700 <= w <= 750]
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
    X.append(spec[selected_band_indices])  # 只選700~750nm
    y.append("Healthy")
    file_names.append(f.name)

# Unhealthy
for f in unhealthy_dir.glob("*.npy"):
    spec = np.load(f)
    X.append(spec[selected_band_indices])  # 只選700~750nm
    y.append("Unhealthy")
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)

# === 3. LDA 降維 ===
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y)

# === 4. 繪圖 ===
plt.figure(figsize=(8, 6))
for label, color in zip(["Healthy", "Unhealthy"], ["blue", "red"]):
    plt.scatter(X_lda[y == label], np.zeros_like(X_lda[y == label]), label=label, alpha=0.7, color=color)

plt.title("LDA Projection (700-750nm)", fontsize=14)
plt.xlabel("LDA Component 1", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = "LDA_Projection_700_750nm.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"LDA 圖片已儲存至 {save_path}")

# === 5. LDA 投影結果保存 ===
lda_df = pd.DataFrame({
    "File": file_names,
    "Label": y,
    "LDA1": X_lda.flatten()
})
lda_df.to_csv("lda_projection_700_750nm.csv", index=False)
print("LDA 投影結果已儲存為 lda_projection_700_750nm.csv")
