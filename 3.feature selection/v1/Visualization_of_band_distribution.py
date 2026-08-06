# 視覺化波段分布
# 用來check為什麼常態那麼少
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro

# === 路徑設定 ===
healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_reflection\npy\healthy")
unhealthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_reflection\npy\unhealthy")

# === 讀取資料 ===
X = []
y = []

for f in healthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Healthy")

for f in unhealthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Unhealthy")

X = np.array(X)
y = np.array(y)


band_to_check = 80

plt.hist(X[y == "Healthy", band_to_check], bins=20, alpha=0.5, label="Healthy", color='blue')
plt.hist(X[y == "Unhealthy", band_to_check], bins=20, alpha=0.5, label="Unhealthy", color='red')
plt.title(f"Band {band_to_check} Distribution")
plt.legend()

# === 儲存圖檔 ===
save_path = f"band_{band_to_check}_distribution.png"
plt.savefig(save_path, dpi=300)
plt.show()

print('finish')
