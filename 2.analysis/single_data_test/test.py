import numpy as np
import spectral
import cv2
from pathlib import Path
import pandas as pd

# === 1. 設定路徑 ===
root = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\test_data0620_S1")
sample_name = "sample1_RT"
day_name = "D0620_S1"

hdr_path = root / f"{sample_name}.hdr"
mask_path = root / f"{day_name}_mask.png"

# === 讀取高光譜影像===
img = spectral.open_image(str(hdr_path))
cube = img.load()  # shape: (rows, cols, bands)

# === 讀取 mask，轉成 boolean mask ===
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
roi_mask = mask > 0

# === 擷取葉片區域的光譜向量 ===
cube_np = np.array(cube)
leaf_pixels = cube_np[roi_mask]

mean_spectrum = np.mean(leaf_pixels, axis=0)
# print(f"平均光譜向量長度: {mean_spectrum.shape[0]}")

# === 輸出 CSV + NPY ===
output_dir = root / "output"
output_dir.mkdir(exist_ok=True)

# 存成 .csv
df = pd.DataFrame(mean_spectrum, columns=["Reflectance"])
df.to_csv(output_dir / f"{sample_name}_mean_spectrum.csv", index_label="Band")

# 存成 .npy
np.save(output_dir / f"{sample_name}_mean_spectrum.npy", mean_spectrum)

print("finished")
