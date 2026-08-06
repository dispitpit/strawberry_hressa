from sklearn.preprocessing import LabelEncoder
from skrebate import ReliefF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
relieff = ReliefF(n_features_to_select=10, n_neighbors=100, discrete_threshold=10,
                  verbose=True, n_jobs=-1)
relieff.fit(X_np, y_encoded)

print("\n=== ReliefF  ===")
# print("label 型態:", relieff._class_type)                 # binary / multiclass / continuous
# print("類別列表  :", relieff._label_list)                  # e.g. ['Healthy','Unhealthy'] after encode
# print("離散門檻  :", relieff.discrete_threshold)
# print("資料型態  :", relieff.data_type)                    # discrete / continuous / mixed
# print("使用的 k  :", relieff.n_neighbors, "(每類 hit/miss 各 k 個)")
# print("缺值數量  :", relieff._missing_data_count)
# print("樣本/特徵 :", relieff._datalen, "/", relieff._num_attributes)

mxlen   = len(str(X_np.shape[1] + 1))
headers = [f"X{str(i).zfill(mxlen)}" for i in range(1, X_np.shape[1] + 1)]
rows = []
for feat, hdr in zip(feat_names, headers):
    t, mx, mn, rng, sd = relieff.attr[hdr]  # (type, max, min, range, std)
    rows.append({"feature": feat, "type": t, "max": mx, "min": mn, "range": rng, "std": sd})
attr_df = pd.DataFrame(rows).set_index("feature")
print("\n=== 特徵型態（前 10 列）===")
print(attr_df[["type"]].head(10))

# ===== 重要度 =====
importances = relieff.feature_importances_
feature_ranking = pd.Series(importances, index=feat_names).sort_values(ascending=False)

# print("\n=== Top 10 VI ===")
print(feature_ranking.head(10))

feature_ranking.head(10).to_csv("top10_vi_relieff_v2.csv", header=["ReliefF Importance"])

# ===== 存檔 =====
save_path = "top10_reliefF_v2.png"
ax = feature_ranking.head(10).plot(kind='barh', figsize=(8, 6), color='orange')
ax.set_xlabel("Importance Values", fontsize=20)
ax.set_title("Top 10 VI by ReliefF", fontsize=24)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

'''
Created distance array in 0.018504619598388672 seconds.
Feature scoring under way ...
Completed scoring in 2.9857017993927 seconds.
=== ReliefF 內部資訊 ===
label 型態: binary
類別列表  : [np.int64(0), np.int64(1)]
離散門檻  : 10
資料型態  : continuous
使用的 k  : 100 (每類 hit/miss 各 k 個)
缺值數量  : 0
樣本/特徵 : 305 / 34
=== 特徵型態（前 10 列）===
               type
feature            
NDVI     continuous
RDVI     continuous
RNDVI    continuous
GI       continuous
SIPI     continuous
MCARI    continuous
MCARI2   continuous
PSRI     continuous
ARI1     continuous
ARI2     continuous
=== Top 10 VI ===
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
'''