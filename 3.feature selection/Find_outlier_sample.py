'''
發現離群樣本數量：4 / 229
離群樣本檔案： [
'D0710_S4_mean_spectrum.npy',
'D0620_S19_mean_spectrum.npy',
'D0625_S19_mean_spectrum.npy',
'D0702_S12_mean_spectrum.npy']
離群樣本已儲存：outlier_samples.csv
'''
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

# === 資料夾設定 ===
healthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0415_LDA\重製1003\ks_npy_train_subset\healthy")
unhealthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0415_LDA\重製1003\ks_npy_train_subset\unhealthy")


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
z_scores = np.abs(zscore(X, axis=0))

# === 判斷是否離群樣本 ===
threshold = 4
outlier_flags = (z_scores > threshold).any(axis=1)
outlier_samples = np.array(file_names)[outlier_flags]

print(f"發現離群樣本數量：{len(outlier_samples)} / {len(file_names)}")
print("離群樣本檔案：", outlier_samples.tolist())
pd.DataFrame({"Outlier Files": outlier_samples}).to_csv("outlier_samples.csv", index=False)
print("離群樣本已儲存：outlier_samples.csv")

print("finish")