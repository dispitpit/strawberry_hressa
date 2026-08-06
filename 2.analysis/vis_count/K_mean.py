# final
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch
from skrebate import ReliefF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ===== 風格 =====
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# ===== 載入資料 =====
df = pd.read_csv("vi_40_output_v2.csv")
X_all = df.drop(columns=["File", "Label"], errors="ignore")
y = df["Label"].astype(str)
y_encoded = LabelEncoder().fit_transform(y)  # 0/1

# 只留數值欄、丟掉含 NaN 的欄
X_numeric  = X_all.select_dtypes(include=[np.number])
X_used     = X_numeric.dropna(axis=1)
feat_names = X_used.columns
X_np       = X_used.to_numpy()

# ===== 執行 ReliefF =====
# === 跑 k=75 與 k=100，取平均 ===
def relieff_scores(k):
    rf = ReliefF(n_features_to_select=10, n_neighbors=k, n_jobs=-1, verbose=False)
    rf.fit(X_np, y_encoded)
    return pd.Series(rf.feature_importances_, index=feat_names)

s75  = relieff_scores(75)
s100 = relieff_scores(100)

# 原始分數取平均
s_mean = (s75 + s100) / 2.0

# 轉為百分比顯示：min–max 正規化到 0–100%
s_pct = (s_mean - s_mean.min()) / (s_mean.max() - s_mean.min() + 1e-12) * 100.0
s_pct = s_pct.sort_values(ascending=False)

# === 輸出 CSV（平均原始分數、百分比分數）===
out_df = pd.DataFrame({
    "score_k75": s75,
    "score_k100": s100,
    "score_mean": s_mean,
    "score_pct_0to100": s_pct
}).loc[s_pct.index]  # 以百分比排序
# out_df.to_csv("relieff_k75_k100_consensus_v2.csv")

# === Acronym → Category 對照 ===
acronym_to_cat = {
    "NDVI": "Structure",
    "RDVI": "Structure",
    "RNDVI": "Structure",
    "RENDVI": "Structure",
    "GI": "Structure",
    "SIPI": "Pigment",
    "MCARI": "Pigment",
    "MCARI2": "Pigment",
    "PSRI": "Pigment",
    "ARI1": "Pigment",
    "ARI2": "Pigment",
    "OSAVI": "Structure",
    "MSR": "Structure",
    "TVI": "Structure",
    "PRI": "Pigment",
    "GNDVI": "Structure",
    "WBI": "Water content",
    "WSCT": "Water content",
    "FRI1": "Physiology",
    "FRI2": "Physiology",
    "FRI3": "Physiology",
    "FRI4": "Physiology",
    "VOG1": "Pigment",
    "VOG2": "Pigment",
    "VOG3": "Pigment",
    "MTCI": "Pigment",
    "EVI": "Structure",
    "CIrededge": "Pigment",
    "RVI": "Structure",
    "MSAVI": "Structure",
    "NDRE": "Structure",
    "CRI1": "Pigment",
    "CRI2": "Pigment",
    "TCARI": "Pigment",
    "DVI": "Structure",
}

# === Category → 顏色（Structure 保持原本橘色）===
cat_to_color = {
    "Structure":       "#607d51",
    "Pigment":         "#c9bf97",
    "Water content":   "#d3dcba",
    "Physiology":      "#849c7d",
}


top20 = s_pct.head(20)
top20_idx = top20.index

colors = []
for acr in top20_idx:
    cat = acronym_to_cat.get(acr, "Structure")
    color = cat_to_color.get(cat, "#FFA500")
    colors.append(color)

# === 畫 Top-20 百分比圖 ===
fig, ax = plt.subplots(figsize=(8, 6))
top20.plot(kind='barh', color=colors, ax=ax)

ax.set_xlabel("Importance (%)", fontsize=20)
ax.set_title("Top 20 VI by ReliefF", fontsize=22)
ax.invert_yaxis()
plt.tight_layout()

# === 加圖例 ===
handles = [Patch(facecolor=c, label=cat) for cat, c in cat_to_color.items()]
ax.legend(handles=handles, title="Category", fontsize=12, title_fontsize=12, loc="lower right")

plt.savefig("top20_reliefF_consensus_pct_v11.png", dpi=300, bbox_inches="tight")
plt.show()
