'''
Train shape: (104, 39), Test shape: (36, 39)
Train label counts: {0: np.int64(52), 1: np.int64(52)}
Test  label counts: {0: np.int64(18), 1: np.int64(18)}
CV(5) Accuracy: 0.8652 ± 0.0470
Test Accuracy : 0.7778
Test Sens     : 0.8333
Test Spec     : 0.7222
Test MCC      : 0.5590
Confusion matrix:
 [[13  5]
 [ 3 15]]
已寫出：svm_vi_low.csv
'''
# -*- coding: utf-8 -*-
"""
Use all VI features from vi_40_output.csv to train SVM.
Random split 3:1 per label (healthy/unhealthy split separately, then combine).
5-fold CV on train, then final test evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

# ====================== 設定 ======================
VI_CSV = Path(r"../../vi_40_low.csv")   # <-- 改成你的實際路徑
TRAIN_RATIO = 0.75
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

OUT_CSV = Path("svm_vi_low.csv")  # 輸出結果表（選用）

# ====================== 讀資料 & 特徵準備 ======================
df = pd.read_csv(VI_CSV)

# 只留下數值特徵；移除 File/Label，並丟掉含 NaN 的欄位
X_all = df.drop(columns=["File", "Label"], errors="ignore")
X_all = X_all.select_dtypes(include=[np.number]).dropna(axis=1)
feat_names = X_all.columns

y_str = df["Label"].astype(str).values
le = LabelEncoder()
y_all = le.fit_transform(y_str)  # 0/1

# ====================== 依 Label 各自「隨機」切 3:1 ======================
def random_split_per_label(df_all: pd.DataFrame, train_ratio=0.75, seed=42):
    rs = np.random.RandomState(seed)
    parts = []
    for label, sub in df_all.groupby("Label", sort=False):
        idx = sub.index.to_numpy()
        rs.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(train_ratio * n)))
        tr_idx = idx[:n_tr]
        te_idx = idx[n_tr:] if n_tr < n else idx[-1:]  # 至少留 1 筆 test
        parts.append(("train", tr_idx))
        parts.append(("test", te_idx))
    # 合併兩類的索引
    train_idx = np.concatenate([idx for tag, idx in parts if tag == "train"])
    test_idx  = np.concatenate([idx for tag, idx in parts if tag == "test"])
    return np.sort(train_idx), np.sort(test_idx)

train_idx, test_idx = random_split_per_label(df[["Label"]], TRAIN_RATIO, RANDOM_STATE)

X_train, X_test = X_all.iloc[train_idx].to_numpy(), X_all.iloc[test_idx].to_numpy()
y_train, y_test = y_all[train_idx], y_all[test_idx]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Train label counts:", dict(pd.Series(y_train).value_counts().sort_index()))
print("Test  label counts:", dict(pd.Series(y_test ).value_counts().sort_index()))

# ====================== 建 SVM(RBF) Pipeline ======================
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),              # z-score（SVM 對尺度很敏感）
    ("clf", SVC(kernel="rbf",
                C=1.0,                         # 你可改/做 GridSearch
                gamma="scale",                 # 預設穩定
                probability=True,              # 要機率才開；不需要可關掉加速
                cache_size=1000,
                shrinking=True,
                random_state=RANDOM_STATE))
])

# ====================== 5-fold CV（在 Train 上） ======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_res = cross_validate(
    svm_pipe, X_train, y_train,
    cv=cv, scoring="accuracy", return_train_score=False, n_jobs=-1
)
cv_mean = float(np.mean(cv_res["test_score"]))
cv_std  = float(np.std(cv_res["test_score"]))
print(f"CV(5) Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

# ====================== 用整個 Train 重訓，評估 Test ======================
svm_pipe.fit(X_train, y_train)
y_pred = svm_pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)  # [[tn, fp],[fn, tp]]
mcc = matthews_corrcoef(y_test, y_pred)

tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
sens = tp / (tp + fn) if (tp + fn) else np.nan
spec = tn / (tn + fp) if (tn + fp) else np.nan

print(f"Test Accuracy : {acc:.4f}")
print(f"Test Sens     : {sens:.4f}")
print(f"Test Spec     : {spec:.4f}")
print(f"Test MCC      : {mcc:.4f}")
print("Confusion matrix:\n", cm)

# ====================== 輸出結果（選用） ======================
pd.DataFrame([{
    "Model": "SVM (RBF, all VIs)",
    "CV5_Acc_Mean": cv_mean,
    "CV5_Acc_Std": cv_std,
    "Test_Accuracy": acc,
    "Test_Sensitivity": sens,
    "Test_Specificity": spec,
    "Test_MCC": mcc,
    "Train_Size": X_train.shape[0],
    "Test_Size": X_test.shape[0],
    "Num_Features": X_train.shape[1]
}]).to_csv(OUT_CSV, index=False)
print(f"已寫出：{OUT_CSV}")
