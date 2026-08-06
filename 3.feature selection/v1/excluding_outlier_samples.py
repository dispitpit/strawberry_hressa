# 已刪除
# 25
# 個離群樣本檔案：
# - D0611_S1_mean_spectrum.npy
# - D0617_S12_mean_spectrum.npy
# - D0617_S1_mean_spectrum.npy
# - D0617_S2_mean_spectrum.npy
# - D0621_S5_mean_spectrum.npy
# - D0628_S3_mean_spectrum.npy
# - D0703_S4_mean_spectrum.npy
# - D0710_S4_mean_spectrum.npy
# - D0619_S10_mean_spectrum.npy
# - D0619_S11_mean_spectrum.npy
# - D0619_S14_mean_spectrum.npy
# - D0619_S15_mean_spectrum.npy
# - D0619_S16_mean_spectrum.npy
# - D0619_S17_mean_spectrum.npy
# - D0619_S18_mean_spectrum.npy
# - D0619_S19_mean_spectrum.npy
# - D0619_S20_mean_spectrum.npy
# - D0620_S19_mean_spectrum.npy
# - D0621_S19_mean_spectrum.npy
# - D0624_S19_mean_spectrum.npy
# - D0625_S19_mean_spectrum.npy
# - D0628_S19_mean_spectrum.npy
# - D0702_S12_mean_spectrum.npy
# - D0709_S20_mean_spectrum.npy
# - D0710_S20_mean_spectrum.npy

import numpy as np
import pandas as pd
from pathlib import Path
import os

# === 資料夾設定 ===
healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy\healthy")
unhealthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy\unhealthy")

# === 讀取離群樣本名單 ===
outlier_df = pd.read_csv("outlier_samples.csv")
outlier_files_set = set(outlier_df["Outlier Files"].tolist())

deleted_files = []

# === 對 healthy 資料夾進行刪除 ===
for f in healthy_dir.glob("*.npy"):
    if f.name in outlier_files_set:
        os.remove(f)
        deleted_files.append(f)

# === 對 unhealthy 資料夾進行刪除 ===
for f in unhealthy_dir.glob("*.npy"):
    if f.name in outlier_files_set:
        os.remove(f)
        deleted_files.append(f)

# === 確認刪除結果 ===
print(f"已刪除 {len(deleted_files)} 個離群樣本檔案：")
for f in deleted_files:
    print(f" - {f.name}")
