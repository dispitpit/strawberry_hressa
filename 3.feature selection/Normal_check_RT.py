'''
=========before=====================
Healthy 波段常態數量: 72 / 300
Unhealthy 波段常態數量: 6 / 300
已儲存詳細結果至 normality_check_results_before.csv
========after=======================
Healthy 波段常態數量: 55 / 300
Unhealthy 波段常態數量: 86 / 300
已儲存詳細結果至 normality_check_results_after.csv
====================================
'''
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro

# === 路徑設定 ===
healthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0415_LDA\重製1003\ks_npy_train_subset\healthy")
unhealthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0415_LDA\重製1003\ks_npy_train_subset\unhealthy")


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

# === 常態檢驗 ===
n_bands = X.shape[1]
results = []

for band_idx in range(n_bands):
    healthy_band_values = X[y == "Healthy", band_idx]
    unhealthy_band_values = X[y == "Unhealthy", band_idx]

    # Shapiro-Wilk test
    p_healthy = shapiro(healthy_band_values).pvalue
    p_unhealthy = shapiro(unhealthy_band_values).pvalue

    results.append({
        "Band": band_idx,
        "Healthy_p": p_healthy,
        "Unhealthy_p": p_unhealthy,
        "Healthy_Normal": p_healthy > 0.05,
        "Unhealthy_Normal": p_unhealthy > 0.05
    })

# === 儲存 ===
df_results = pd.DataFrame(results)
df_results.to_csv("normality_check_results_after.csv", index=False)
healthy_normal_count = df_results["Healthy_Normal"].sum()
unhealthy_normal_count = df_results["Unhealthy_Normal"].sum()

print(f"Healthy 波段常態數量: {healthy_normal_count} / {n_bands}")
print(f"Unhealthy 波段常態數量: {unhealthy_normal_count} / {n_bands}")
print("已儲存詳細結果至 normality_check_results_after.csv")
