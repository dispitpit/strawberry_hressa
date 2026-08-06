
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
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from joblib import dump

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

# ====================== 設定 ======================
VI_CSV = Path(r"vi_40_low_v2.csv")
TRAIN_RATIO = 0.75
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_CSV = Path("result/2_low/vi_low.csv")

# ====================== 讀資料 & 特徵準備 ======================
df = pd.read_csv(VI_CSV)

X_all = df.drop(columns=["File", "Label"], errors="ignore")
X_all = X_all.select_dtypes(include=[np.number]).dropna(axis=1)
feat_names = X_all.columns

y_str = df["Label"].astype(str).values
le = LabelEncoder()
y_all = le.fit_transform(y_str)  # 0/1

# ====================== 依 Label 各自隨機切 3:1 ======================
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

# ====================== 模型（Pipeline + Scaler） ======================
'''
pipline[]: 避免資料外漏  [做cv->train->analysis] 
'''
def build_models():
    models = {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()), # 對每個特徵z-score
            ("clf", SVC(kernel="rbf",  # 擬合非線性邊界
                        C = 3, # 正則強度反比 [0.1, 1, 3, 10, 30]
                        gamma = 'scale', # ["scale", 0.1, 0.03, 0.01, 0.003]
                        probability=True, # 機率估計
                        cache_size=1000, # MB，特徵多就調大
                        shrinking=True, # 預設TRUE
                        random_state=RANDOM_STATE # 隨機種子42
                        ))]),

        "k-NN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,  # 近鄰數 K，影響平滑度 [3,5,7,9,11,15]
                weights='uniform',  # 距離加權方式：'uniform' 同權 / 'distance' 近者權重較大
                algorithm='auto',  # 鄰居搜尋演算法：'auto' | 'ball_tree' | 'kd_tree' | 'brute'
                leaf_size=30,  # 樹結構葉節點大小（影響速度/記憶體；ball_tree/kd_tree 用到）
                metric='minkowski',  # 距離度量：常用 'minkowski'（搭配 p=2 等於歐式距離）
                p=2,  # Minkowski 的 p；p=2 → Euclidean，p=1 → Manhattan
                metric_params=None,  # 自訂距離的參數（通常用不到）
                n_jobs=-1  # 並行查詢（brute/距離計算時有效；Windows 下也支援）
            ))]),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",  # 正則化種類：'l2' 常用；'l1'/'elasticnet' 需搭配合適 solver
                C=1.0,  # 正則強度的反比，越大越貼訓練資料；常試 [0.1, 1, 3, 10]
                solver="lbfgs",  # 求解器：'lbfgs'(預設, 支援多類&l2) / 'liblinear'(小資料,l1/l2)
                                #        'saga'(大資料/稀疏, 支援 l1/elasticnet) / 'sag'
                l1_ratio=None,  # 僅在 penalty='elasticnet' 且 solver='saga' 時使用（0~1）
                fit_intercept=True,  # 是否學 intercept（偏置）
                class_weight=None,  # 類別不平衡可用 'balanced' 或自訂 {0:w0, 1:w1}
                max_iter=1000,  # 最大迭代次數（收斂不足時調大）
                tol=1e-4,  # 收斂容忍度（調小更精準但更慢）
                # multi_class="auto",  # 多類策略：'auto'；或 'ovr'、'multinomial'(需 lbfgs/saga)
                n_jobs=-1,  # 並行（對 'liblinear'/'saga' 等有效；'lbfgs' 不用此參數）
                random_state=RANDOM_STATE  # 部分 solver（sag/saga）與打散流程會用到
            ))]),

        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(
                solver="lsqr",  # 'svd'(預設, 快速且不需估共變矩陣) /
                               # 'lsqr' 或 'eigen'（可搭 shrinkage，對高維更穩）
                shrinkage=None,  # None(不縮減) / 'auto'(Ledoit–Wolf) / 0~1 浮點（需 lsqr/eigen）
                n_components=None,  # 降到 <=(類別數-1) 維；None 表示保留全部判別軸
                priors=None,  # 類別先驗機率；None 用數據頻率
                tol=1e-4,  # 收斂容忍度（svd 用）
                store_covariance=False  # 是否存共變矩陣（分析用）
            ))]),

        # 樹類本不需縮放；為統一流程保留，且 with_mean=False 省記憶體
        "Random Forest": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", RandomForestClassifier(
                n_estimators=300,  # 樹數量；更多更穩但更慢。常試 [100, 300, 500]
                criterion="gini",  # 'gini' / 'entropy' / 'log_loss'(新款)
                max_depth=None,  # 限制樹深度避免過擬合；None 表示不限制
                min_samples_split=2,  # 節點再切分所需最小樣本
                min_samples_leaf=1,  # 葉節點最少樣本；>1 可抑制過擬合（如 2, 3, 5）
                max_features="sqrt",  # 每次切分抽取的特徵數：'sqrt' 常用；也可 'log2'、None 或比例
                bootstrap=True,  # 是否自助抽樣；配合 oob_score 可估 OOB 準確率
                oob_score=False,  # 想看 OOB 就設 True（需 bootstrap=True）
                class_weight=None,  # 不平衡時可 'balanced' 或自訂 {0:w0,1:w1}
                n_jobs=-1,  # 平行化
                random_state=RANDOM_STATE
            ))]),

        # XGBoost（新版相容；無 use_label_encoder）
        "XGBoost": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", XGBClassifier(
                n_estimators=300,  # 樹數量（弱學習器個數）→ 越多越穩但越慢；可配合 early_stopping
                max_depth=4,  # 單棵樹最大深度；大易過擬合、太小欠擬合，常試 [3,4,5,6]
                learning_rate=0.1,  # 步長（eta）；小步長+更多樹通常泛化更好（如 0.05, 0.1, 0.2）
                subsample=0.8,  # 行抽樣比例（樣本抽樣）；抑制過擬合，常試 [0.6~1.0]
                colsample_bytree=0.8,  # 列抽樣比例（特徵抽樣）；常試 [0.6~1.0]
                objective="binary:logistic",  # 二元分類輸出為概率
                eval_metric="logloss",  # 評估指標：logloss / auc（不平衡時常用 auc）
                reg_lambda=1.0,  # L2 正則（lambda），默認 1；可調 [0, 1, 5, 10]
                reg_alpha=0.0,  # L1 正則（alpha），可調 [0, 0.1, 1]
                min_child_weight=1.0,  # 最小葉子節點樣本權重和（控制複雜度），常試 [1,3,5,10]
                gamma=0.0,  # 節點分裂所需的最小損失減少量（>=0）；大→更保守
                scale_pos_weight=1.0,  # 類別不平衡時用；約等於 Neg/Pos 比
                tree_method="hist",  # CPU 快速直方圖演算法；有 GPU 時可改 "gpu_hist"
                n_jobs=-1,  # 多執行緒
                random_state=RANDOM_STATE
            ))])
    }
    return models
# ====================== 多模型：5-fold CV + 最終測試 ======================
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
import numpy as np
import pandas as pd

models = build_models()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

all_rows = []
confmats = {}

for name, pipe in models.items():
    print(f"\n=== {name} ===")

    # ---- 5-fold CV on TRAIN
    cv_res = cross_validate(
        pipe, X_train, y_train,
        cv=cv, scoring="accuracy",
        return_train_score=False, n_jobs=-1
    )
    cv_mean = float(np.mean(cv_res["test_score"]))
    cv_std  = float(np.std(cv_res["test_score"]))
    print(f"CV(5) Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
    else:
        sens = spec = np.nan

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Sens     : {sens:.4f}")
    print(f"Test Spec     : {spec:.4f}")
    print(f"Test MCC      : {mcc:.4f}")
    print("Confusion matrix:\n", cm)

    cm_path = OUT_CSV.with_name(f"confmat_{name.replace(' ','_').replace('(','').replace(')','')}.csv")
    pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"]).to_csv(cm_path, index=True)

    all_rows.append({
        "Model": name,
        "CV5_Acc_Mean": cv_mean,
        "CV5_Acc_Std":  cv_std,
        "Test_Accuracy": acc,
        "Test_Sensitivity": sens,
        "Test_Specificity": spec,
        "Test_MCC": mcc,
        "ConfMat_CSV": str(cm_path),
        "Train_Size": X_train.shape[0],
        "Test_Size":  X_test.shape[0],
        "Num_Features": X_train.shape[1],
    })
    confmats[name] = cm

result_df = pd.DataFrame(all_rows).sort_values(by="Test_MCC", ascending=False)
result_df.to_csv(OUT_CSV, index=False)
print(f"\n output：{OUT_CSV}")
print(result_df[["Model","CV5_Acc_Mean","Test_Accuracy","Test_Sensitivity","Test_Specificity","Test_MCC"]])
