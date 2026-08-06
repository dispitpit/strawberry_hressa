# fake and real
import numpy as np
import torch
import os
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# file
# trueB是300bands、fakeB是32bands
TRAINB_DIR = Path(
    r"D:\Amanda_strawberry\pytorch-CycleGAN-and-pix2pix-master\datasets\strawberry_HSI\trainB"
)
FAKEB_DIR = Path(
    r"D:\Amanda_strawberry\pytorch-CycleGAN-and-pix2pix-master\final_result\low_output_v1\trainA_fakeB"
)
WAVE_PATH = Path(
    r"D:\Amanda_strawberry\pytorch-CycleGAN-and-pix2pix-master\test_models\wavelengths.npy"
)

SELECTED_BANDS = sorted(
    {30, 31, 32, 33, 34, 35, 60, 61, 87,
     96, 103, 110, 121, 131, 134, 152,
     154, 155, 159, 169, 173, 174, 180, 1,
     90, 191, 194, 207, 222, 223, 225, 231, 273})

# _________________________________________________________________
def to_chw(arr):
    if arr.ndim != 3:
        raise ValueError(arr.shape)
    if arr.shape[0] <= 400:
        return arr
    return arr.transpose(2, 0, 1)

def spectrum_mean(cube_chw):
    # 每個 band 對整張圖取平均
    return cube_chw.reshape(cube_chw.shape[0], -1).mean(axis=1)

# _________________________________________________________________

# ============================
# 讀波長
# ============================
wavelengths = np.load(WAVE_PATH).astype(np.float32)
wave_sel = wavelengths[SELECTED_BANDS]

records = []

# ============================
# trueB（藍）
# ============================
for f in sorted(TRAINB_DIR.glob("*.npy")):
    cube = to_chw(np.load(f).astype(np.float32))
    cube_sel = cube[SELECTED_BANDS]
    spec = spectrum_mean(cube_sel)

    for wl, val in zip(wave_sel, spec):
        records.append({
            "wavelength": wl,
            "reflectance": val,
            "type": "trueB",
            "file": f.name
        })

# ============================
# fakeB（紅）
# ============================
for f in sorted(FAKEB_DIR.glob("*.npy")):
    cube = to_chw(np.load(f).astype(np.float32))

    if cube.shape[0] != len(SELECTED_BANDS):
        raise ValueError(f"{f.name} channel mismatch: {cube.shape}")

    spec = spectrum_mean(cube)

    for wl, val in zip(wave_sel, spec):
        records.append({
            "wavelength": wl,
            "reflectance": val,
            "type": "fakeB",
            "file": f.name
        })

df = pd.DataFrame(records)

# ============================
# 繪圖
# ============================
plt.figure(figsize=(11, 6))
sns.lineplot(
    data=df,
    x="wavelength",
    y="reflectance",
    hue="type",
    units="file",
    estimator=None,
    palette={"trueB": "blue", "fakeB": "red"},
    alpha=0.25,
    linewidth=1.2
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Spectral Curves: trueB (blue) vs fakeB (red)")
plt.grid(True, linestyle="--", alpha=0.3)

plt.legend(title=None)
plt.tight_layout()
plt.savefig("trueB_vs_fakeB_all_samples.png", dpi=200)
plt.show()