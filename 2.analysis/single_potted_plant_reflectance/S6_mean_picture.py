import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 載入實際波長 ===
wavelengths = np.load("wavelengths.npy")  # shape = (300,)
# === 兩類資料夾 ===
healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\S6\healthy")
unhealthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\S6\unhealthy")
# === 輸出圖檔儲存位置 ===
save_path = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\S6\mean_spectrum_plot_for10.png")
# === 畫圖 ===
plt.figure(figsize=(12, 6))

# === Healthy ===
for npy_file in sorted(healthy_dir.glob("*.npy")):
    spectrum = np.load(npy_file)
    if len(spectrum) == 300:
        plt.plot(wavelengths, spectrum, color='blue', alpha=0.4, linewidth=1)

# === Unhealthy ===
for npy_file in sorted(unhealthy_dir.glob("*.npy")):
    spectrum = np.load(npy_file)
    if len(spectrum) == 300:
        plt.plot(wavelengths, spectrum, color='red', alpha=0.4, linewidth=1)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Mean Spectra: Healthy (Blue) vs Unhealthy (Red)")
plt.grid(True)
plt.tight_layout()

# === 儲存 + 顯示 ===
plt.savefig(save_path, dpi=300)
plt.show()

print("finish")