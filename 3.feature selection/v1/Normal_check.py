# =============test_0415===========================
# Healthy 波段常態數量: 148 / 300
# Unhealthy 波段常態數量: 216 / 300
# 已儲存詳細結果至 normality_check_results0415.csv

# =============test_0322===========================
# Healthy 波段常態數量: 66 / 300
# Unhealthy 波段常態數量: 4 / 300
# 已儲存詳細結果至 normality_check_results0322.csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import shapiro

# === 路徑設定 ===
healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy_LDA\healthy")
unhealthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy_LDA\unhealthy")

# === 讀取資料 ===
X = []
y = []

for f in healthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Healthy")

for f in unhealthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append("Unhealthy")

X = np.array(X)  # shape: (n_samples, n_bands)
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

# === 結果儲存 ===
df_results = pd.DataFrame(results)
df_results.to_csv("normality_check_results_0415.csv", index=False)

# === 總結 ===
healthy_normal_count = df_results["Healthy_Normal"].sum()
unhealthy_normal_count = df_results["Unhealthy_Normal"].sum()

print(f"Healthy 波段常態數量: {healthy_normal_count} / {n_bands}")
print(f"Unhealthy 波段常態數量: {unhealthy_normal_count} / {n_bands}")
print("已儲存詳細結果至 normality_check_results_before.csv")
