# SPA前處理
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# === 路徑 ===
healthy_dir = r'C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy_SPA\healthy'
unhealthy_dir = r'C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy_SPA\unhealthy'

# === 讀所有 .npy ===
X = []
y = []

# 讀 healthy
for file in os.listdir(healthy_dir):
    if file.endswith('.npy'):
        data = np.load(os.path.join(healthy_dir, file))
        X.append(data)
        y.append(0)

# 讀 unhealthy
for file in os.listdir(unhealthy_dir):
    if file.endswith('.npy'):
        data = np.load(os.path.join(unhealthy_dir, file))
        X.append(data)
        y.append(1)

X = np.array(X)
y = np.array(y)

Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.4, random_state=0)

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
Xcal = min_max_scaler.fit_transform(Xcal)
Xval = min_max_scaler.transform(Xval)

print("Xcal shape:", Xcal.shape)
print("Xval shape:", Xval.shape)
print("ycal:", ycal[:5])
print("yval:", yval[:5])
