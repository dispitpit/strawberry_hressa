# npy
# Train: X=(229, 300), Test: X=(76, 300)
# low
# Train: X=(104, 300), Test: X=(36, 300)
# high
# Train: X=(150, 300), Test: X=(50, 300)

"""
Use existing KS split lists -> load raw spectra -> 5-fold CV on train -> final test eval.
"""
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
CONCENTRATION = "high"  # 對應 {BASE_DIR}/{healthy,unhealthy}
BASE_DIR = Path(fr"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/{CONCENTRATION}")

SPLIT_DIR = Path(f"ks_{CONCENTRATION}_divided_result")
TRAIN_LIST = SPLIT_DIR / f"ks_split_train_{CONCENTRATION}.csv"
TEST_LIST  = SPLIT_DIR / f"ks_split_test_{CONCENTRATION}.csv"
'''
KS的依據: vi_40_output.csv -> ["VOG1", "NDRE", "MTCI", "RENDVI", "MCARI"]
'''
# ========================輸出==================
OUT_DIR = Path("ML_result/3_high")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CSV = OUT_DIR / f"ml_ks_cv_{CONCENTRATION}_v2.csv"

# 隨機種子: RandomForest、XGBoost、SVM用
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ====================== 模型 ======================
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
# ====================== 載入資料 ======================
def load_list_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), f"找不到清單：{path}"
    df = pd.read_csv(path)
    need = {"File", "Label"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path} 缺少欄位：{miss}")
    return df[["File", "Label"]].copy()

def load_raw_spectra_by_list(df: pd.DataFrame, base_dir: Path, label_encoder: LabelEncoder):
    specs, labels, names, missing = [], [], [], []
    for _, r in df.iterrows():
        f = str(r["File"])
        lab = str(r["Label"])
        p = base_dir / lab.lower() / f
        if p.exists():
            specs.append(np.load(p))
            labels.append(lab)
            names.append(f)
        else:
            missing.append(str(p))
    if missing:
        warnings.warn(f"找不到以下檔案，共 {len(missing)} 筆（將略過）：\n" +
                      "\n".join(missing[:12]) + ("..." if len(missing) > 12 else ""))
    X = np.array(specs)
    y = label_encoder.transform(labels) if labels else np.array([])
    return X, y, names

# ====================== 評估流程 ======================
def eval_with_cv_and_test(models: dict, X_train, y_train, X_test, y_test, out_dir: Path):
    # 5倍cv
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, pipe in models.items():
        print(f"\n=== {name} ===")
        # 5-fold CV on TRAIN
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring="accuracy",
                                return_train_score=False, n_jobs=-1)
        cv_mean = float(np.mean(cv_res["test_score"]))
        cv_std  = float(np.std(cv_res["test_score"]))
        print(f"CV(5) Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        # Refit on all TRAIN, then evaluate on TEST
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred) if len(y_test) else np.nan
        cm = confusion_matrix(y_test, y_pred) if len(y_test) else np.array([[0,0],[0,0]])
        mcc = matthews_corrcoef(y_test, y_pred) if len(y_test) else np.nan
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
        else:
            sensitivity = specificity = np.nan

        print(f"Test Accuracy : {acc:.4f}" if not np.isnan(acc) else "Test Accuracy : N/A")
        print(f"Test Sens     : {sensitivity:.4f}" if not np.isnan(sensitivity) else "Test Sens     : N/A")
        print(f"Test Spec     : {specificity:.4f}" if not np.isnan(specificity) else "Test Spec     : N/A")
        print(f"Test MCC      : {mcc:.4f}" if not np.isnan(mcc) else "Test MCC      : N/A")

        # 存混淆矩陣
        cm_path = out_dir / f"confmat_{name.replace(' ','_').replace('(','').replace(')','')}_{CONCENTRATION}.csv"
        pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"]).to_csv(cm_path, index=True)

        rows.append({
            "Model": name,
            "CV5_Acc_Mean": cv_mean,
            "CV5_Acc_Std": cv_std,
            "Test_Accuracy": acc,
            "Test_Sensitivity": sensitivity,
            "Test_Specificity": specificity,
            "Test_MCC": mcc,
            "ConfMat_CSV": str(cm_path)
        })

    return pd.DataFrame(rows)

# ====================== 主程式 ======================
def main():
    # 取原來的資料
    healthy_dir = BASE_DIR / "healthy"
    unhealthy_dir = BASE_DIR / "unhealthy"
    assert healthy_dir.exists() and unhealthy_dir.exists(), f"找不到 {healthy_dir} 或 {unhealthy_dir}"

    # 讀清單
    train_list = load_list_csv(TRAIN_LIST)
    test_list  = load_list_csv(TEST_LIST)

    # 建立 LabelEncoder（用 train+test 的標籤）
    le = LabelEncoder().fit(pd.concat([train_list["Label"], test_list["Label"]], axis=0).astype(str))

    # 載入原始光譜
    X_train, y_train, train_names = load_raw_spectra_by_list(train_list, BASE_DIR, le)
    X_test,  y_test,  test_names  = load_raw_spectra_by_list(test_list,  BASE_DIR, le)

    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("訓練或測試資料為空，請檢查清單與檔案路徑。")
    print(f"Train: X={X_train.shape}, Test: X={X_test.shape}")

    # 建模 + CV + Test
    models = build_models()
    result_df = eval_with_cv_and_test(models, X_train, y_train, X_test, y_test, OUT_DIR)

    # 儲存總表
    result_df.to_csv(RESULT_CSV, index=False)
    print(f"\n已寫出：{RESULT_CSV}")

    # 保存模型
    dump(models["SVM (RBF)"], f"best_model_{CONCENTRATION}.joblib")
    print("模型已儲存為 best_model.joblib")

if __name__ == "__main__":
    main()
