# 發現離群樣本數量：25 / 305
# 離群樣本檔案： ['D0611_S1_mean_spectrum.npy', 'D0617_S12_mean_spectrum.npy', 'D0617_S1_mean_spectrum.npy', 'D0617_S2_mean_spectrum.npy', 'D0621_S5_mean_spectrum.npy', 'D0628_S3_mean_spectrum.npy', 'D0703_S4_mean_spectrum.npy', 'D0710_S4_mean_spectrum.npy', 'D0619_S10_mean_spectrum.npy', 'D0619_S11_mean_spectrum.npy', 'D0619_S14_mean_spectrum.npy', 'D0619_S15_mean_spectrum.npy', 'D0619_S16_mean_spectrum.npy', 'D0619_S17_mean_spectrum.npy', 'D0619_S18_mean_spectrum.npy', 'D0619_S19_mean_spectrum.npy', 'D0619_S20_mean_spectrum.npy', 'D0620_S19_mean_spectrum.npy', 'D0621_S19_mean_spectrum.npy', 'D0624_S19_mean_spectrum.npy', 'D0625_S19_mean_spectrum.npy', 'D0628_S19_mean_spectrum.npy', 'D0702_S12_mean_spectrum.npy', 'D0709_S20_mean_spectrum.npy', 'D0710_S20_mean_spectrum.npy']
# 離群樣本已儲存：outlier_samples.csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

# === 資料夾設定 ===
healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_reflection\npy\healthy")
unhealthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_reflection\npy\unhealthy")

# === 載入資料 ===
X = []
y = []
file_names = []

for f in healthy_dir.glob("*.npy"):
    spec = np.load(f)
    X.append(spec)
    y.append("Healthy")
    file_names.append(f.name)

for f in unhealthy_dir.glob("*.npy"):
    spec = np.load(f)
    X.append(spec)
    y.append("Unhealthy")
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)

# === 計算每個樣本的 Z-score ===
z_scores = np.abs(zscore(X, axis=0))  # shape: (n_samples, n_bands)

# === 判斷是否離群樣本 ===
threshold = 3  # Z-score 超過 3 視為離群
outlier_flags = (z_scores > threshold).any(axis=1)  # 只要有一個波段是離群，就當作離群樣本

# === 匯出離群樣本 ===
outlier_samples = np.array(file_names)[outlier_flags]

print(f"發現離群樣本數量：{len(outlier_samples)} / {len(file_names)}")
print("離群樣本檔案：", outlier_samples.tolist())

# 儲存離群樣本列表
pd.DataFrame({"Outlier Files": outlier_samples}).to_csv("outlier_samples.csv", index=False)
print("離群樣本已儲存：outlier_samples.csv")
