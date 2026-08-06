from sklearn.preprocessing import LabelEncoder
from skrebate import ReliefF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# ===== 載入資料 =====
df = pd.read_csv("vi_40_output_v2.csv")
X_all = df.drop(columns=["File", "Label"], errors="ignore")
y = df["Label"].astype(str)
y_encoded = LabelEncoder().fit_transform(y)  # 0/1

# 只留數值欄
X_numeric  = X_all.select_dtypes(include=[np.number])
X_used     = X_numeric.dropna(axis=1)
feat_names = X_used.columns
X_np       = X_used.to_numpy()

# ===== 執行 ReliefF =====
# ========= k 候選與合法性檢查 =========
ks = [25, 50, 75, 100, 125, 150, 175]
cls_cnt = np.bincount(y_encoded)
minority = int(cls_cnt.min())
valid_ks = [k for k in ks if k <= minority]
if len(valid_ks) < len(ks):
    print(f"[警告] 少數類樣本數 = {minority}，已自動過濾非法 k: {set(ks) - set(valid_ks)}")
# print("使用的 k 清單：", valid_ks)

# ========= 跑不同 k 的 ReliefF =========
def run_relieff(k: int) -> pd.Series:
    rf = ReliefF(n_features_to_select=10, n_neighbors=k, discrete_threshold=10, n_jobs=-1, verbose=False)
    rf.fit(X_np, y_encoded)
    s = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    s.to_csv(f"relieff_k{k}_raw.csv", header=["score"])
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    s_norm.to_csv(f"relieff_k{k}_norm01.csv", header=["score_norm01"])
    return s

scores_by_k = {k: run_relieff(k) for k in valid_ks}

# ========= 穩定度：Top-10 重疊 + Spearman 名次相關 =========
print("\n=== 穩定度（Top-10 overlap / Spearman rho）===")
for i, ki in enumerate(valid_ks):
    for kj in valid_ks[i+1:]:
        s1, s2 = scores_by_k[ki], scores_by_k[kj]
        # Top-10 重疊
        top1 = set(s1.head(10).index)
        top2 = set(s2.head(10).index)
        overlap = len(top1 & top2)
        # Spearman（整體名次）
        order = list(feat_names)  # 原始特徵順序
        v1 = scores_by_k[ki].reindex(order).to_numpy()
        v2 = scores_by_k[kj].reindex(order).to_numpy()
        rho, _ = spearmanr(v1, v2)  # 用原始分數
        # rho, _ = spearmanr(s1.rank(ascending=False), s2.rank(ascending=False))
        print(f"k={ki} vs {kj}: Top10 overlap={overlap}/10, Spearman={rho:.3f}")

# ========= 以最大 k 畫 Top-10 圖 =========
base_k = max(valid_ks)
base_scores = scores_by_k[base_k]
print(f"\n=== Top 10 @ k={base_k} ===")
print(base_scores.head(10))

ax = base_scores.head(10).plot(kind='barh', figsize=(8, 6), color='orange')
ax.set_xlabel("Importance Values", fontsize=20)
ax.set_title(f"Top 10 VI by ReliefF (k={base_k})", fontsize=24)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"top10_reliefF_k{base_k}.png", dpi=300, bbox_inches="tight")
plt.show()

# ========= 共識分數 =========
# 先對齊所有特徵，缺者補 0（一般都齊）
df_scores = pd.DataFrame({f"k{k}": s for k, s in scores_by_k.items()}).fillna(0.0)
df_scores["consensus_mean"] = df_scores.mean(axis=1)
df_scores.sort_values("consensus_mean", ascending=False, inplace=True)
df_scores.to_csv("relieff_consensus_mean_v2.csv")

print("\n=== Top 10 by consensus (mean over ks) ===")
print(df_scores["consensus_mean"].head(10))

ax = df_scores["consensus_mean"].head(10).plot(kind='barh', figsize=(8, 6), color='salmon')
ax.set_xlabel("Consensus Importance (Mean over k)", fontsize=20)
ax.set_title("Top 10 VI by ReliefF (Consensus over k)", fontsize=24)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("top10_reliefF_consensus_mean_v2.png", dpi=300, bbox_inches="tight")
plt.show()

'''
[警告] 少數類樣本數 = 103，已自動過濾非法 k: {125, 150, 175}
使用的 k 清單： [25, 50, 75, 100]
=== 穩定度（Top-10 overlap / Spearman rho）===
k=25 vs 50: Top10 overlap=9/10, Spearman=0.919
k=25 vs 75: Top10 overlap=8/10, Spearman=0.878
k=25 vs 100: Top10 overlap=6/10, Spearman=0.761
k=50 vs 75: Top10 overlap=8/10, Spearman=0.965
k=50 vs 100: Top10 overlap=7/10, Spearman=0.902
k=75 vs 100: Top10 overlap=8/10, Spearman=0.958
=== Top 10 @ k=100 ===
MTCI      0.041262
NDRE      0.041211
VOG1      0.041154
RNDVI     0.031749
RENDVI    0.031749
MCARI     0.031243
MCARI2    0.030138
RVI       0.023885
VOG3      0.020127
ARI1      0.019161
dtype: float64
=== Top 10 by consensus (mean over ks) ===
NDRE      0.030183
VOG1      0.029824
MTCI      0.029295
RNDVI     0.023945
RENDVI    0.023945
RVI       0.022185
MCARI     0.022022
FRI1      0.021477
MCARI2    0.020847
WBI       0.018437
Name: consensus_mean, dtype: float64
'''