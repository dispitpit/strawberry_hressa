import os
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

from datetime import datetime
# from zoneinfo import ZoneInfo  # 標準時間用的
# ===============================
# 使用者設定區
# ===============================
concentration = "high"
vi_features = ["MTCI", "VOG3", "VOG1", "Chlorophyll Index RedEdge", "CInir"]  # 目前只用來抓檔名與標籤

# 你的資料路徑
vi_csv_path = Path(rf"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/vi_40_output.csv")
base_dir    = Path(rf"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/{concentration}")
healthy_dir = base_dir / "healthy"
unhealthy_dir = base_dir / "unhealthy"

# wavelengths.npy 路徑（請改成你的）
wavelengths_npy_path = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/wavelengths.npy")

# 要跑哪些波段組合（名稱需對應 band_sets_nm 的 key）
# bandset_names_to_run = ["LDA", "VIs", "PCA", "SPA", "RF", "CARS+LASSO"]
bandset_names_to_run = ["ALL"]

# 允許的目標波長與實際波長的最大差距（nm）。若超過會印出警告，但仍取最近者。
max_delta_nm = 5.0

# ===============================
# 定義各方法選出的「目標波長」(nm)
# ===============================
band_sets_nm = {
    "LDA": [673.59, 655.29, 641.02, 647.14, 618.52],
    "VIs": [754.36, 709.01, 681.70, 747.31, 740.27],
    "PCA": [1001.59, 997.53, 991.45, 993.48],   # 你標註 (Inaccurate)，仍提供對映
    "SPA": [451.17],
    "RF":  [405.86, 401.50, 410.20, 403.68, 421.04],
    "CARS+LASSO": [433.99, 442.59, 519.01, 645.10, 710.03],
    "ALL": [673.59, 655.29, 641.02, 647.14, 618.52, 754.36, 709.01, 681.70, 747.31, 740.27, 1001.59, 997.53, 991.45, 993.48, 451.17, 405.86, 401.50, 410.20, 403.68, 421.04, 433.99, 442.59, 519.01, 645.10, 710.03 ]
}
# band_sets_nm = {
#     "ALL": [673.59, 655.29, 641.02, 647.14, 618.52, 754.36, 709.01, 681.70, 747.31, 740.27, 1001.59, 997.53, 991.45, 993.48, 451.17, 405.86, 401.50, 410.20, 403.68, 421.04, 433.99, 442.59, 519.01, 645.10, 710.03 ]
# }

# ===============================
# 模型定義（與你原本相同）
# ===============================
models = {
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

os.makedirs("all_result", exist_ok=True)

# ===============================
# 工具函式：把目標波長對應到實際 band index
# ===============================
def map_targets_to_indices(target_nm_list, wavelengths, max_delta_nm=5.0):
    """
    將目標波長（浮點數清單）映射到 wavelengths 陣列中「最接近的索引」。
    回傳:
      indices: List[int]，與 target_nm_list 同順序
      mapping_df: 對照表 DataFrame（包含 target_nm、matched_nm、index、delta_nm、flag）
    """
    indices = []
    rows = []
    for t in target_nm_list:
        # 找到與 t 差的最小的波長索引
        idx = int(np.argmin(np.abs(wavelengths - t)))
        matched_nm = float(wavelengths[idx])
        delta = abs(matched_nm - t)
        flag = "" if delta <= max_delta_nm else "WARN>Δ>max_delta"
        indices.append(idx)
        rows.append({
            "target_nm": t,
            "matched_nm": matched_nm,
            "index": idx,
            "delta_nm": delta,
            "flag": flag
        })
    mapping_df = pd.DataFrame(rows)
    return indices, mapping_df

# ===============================
# 載入 .npy 光譜，並可在載入時只取特定 band（子抽樣）
# ===============================
def load_raw_spectra(df, base_dir, label_encoder, band_indices=None):
    """
    df: 包含 File, Label 欄位；base_dir: high/ 底下資料夾；label_encoder: 已 fit 的 LabelEncoder
    band_indices: List[int] 或 None。None=取全頻段；否則只取指定索引。
    """
    specs, labels, names = [], [], []
    for _, row in df.iterrows():
        fname = row["File"]
        label = row["Label"]
        fpath = (base_dir / label.lower() / fname)
        if fpath.exists():
            spec = np.load(fpath)  # 期望 shape=(n_bands,)；若為影像/矩陣，會自動取空間平均
            if spec.ndim > 1:
                # 若是 HxW x C 或 HxC，將空間軸做平均，保留最後一軸為光譜
                spatial_axes = tuple(range(spec.ndim - 1))
                spec = spec.mean(axis=spatial_axes)
            # 子抽樣指定 band
            if band_indices is not None:
                spec = spec[band_indices]
            specs.append(spec.astype(np.float32))
            labels.append(label)
            names.append(fname)
    return np.array(specs), label_encoder.transform(labels), names

# ===============================
# 讀取 VI CSV：只用它來取得共同檔名與標籤
# ===============================
vi_df = pd.read_csv(vi_csv_path)
selected_files = {f.name for f in healthy_dir.glob("*.npy")} | {f.name for f in unhealthy_dir.glob("*.npy")}
vi_subset_df = vi_df[vi_df["File"].isin(selected_files)].copy()

label_encoder = LabelEncoder()
vi_subset_df["EncodedLabel"] = label_encoder.fit_transform(vi_subset_df["Label"])

# 保持「同一份 random split」以利不同 band set 公平比較
train_df, test_df = train_test_split(
    vi_subset_df,
    test_size=0.25,
    stratify=vi_subset_df["EncodedLabel"],
    random_state=42
)

# 讀 wavelengths.npy
wavelengths = np.load(wavelengths_npy_path).astype(float)  # shape: (n_bands,)

# ==========================================
# 逐一跑每個「波段組合」
# ==========================================
# bandset_anmes_to_run: 設定的組合
for bandset_name in bandset_names_to_run:
    target_nm_list = band_sets_nm[bandset_name]
    band_indices, mapping_df = map_targets_to_indices(target_nm_list, wavelengths, max_delta_nm=max_delta_nm)

    # 顯示對照（目標 → 實際）
    print(f"\n=== Band set: {bandset_name} ===")
    for _, r in mapping_df.iterrows():
        warn = " error" if r["flag"] else ""
        print(f" target {r['target_nm']:.2f} nm -> idx {int(r['index'])} (actual {r['matched_nm']:.2f} nm, Δ={r['delta_nm']:.2f} nm){warn}")

    # 將對照表存檔
    mapping_out = Path(f"all_result/{bandset_name}_band_mapping.csv")
    mapping_df.to_csv(mapping_out, index=False)

    # ---------- Random Split ----------
    X_train, y_train, train_names = load_raw_spectra(train_df, healthy_dir.parent, label_encoder, band_indices=band_indices)
    X_test,  y_test,  test_names  = load_raw_spectra(test_df,  healthy_dir.parent, label_encoder, band_indices=band_indices)

    random_results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        mcc = matthews_corrcoef(y_test, y_pred)

        # 二分類時取出 TN,FP,FN,TP；若不是二分類，這裡做保護（但你的情境是 binary）
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
        else:
            sensitivity = specificity = np.nan

        print(f"[{bandset_name} | Random] {name}: SEN={sensitivity:.4f}, SPE={specificity:.4f}, ACC={acc:.4f}, MCC={mcc:.4f}")

        random_results.append({
            "BandSet": bandset_name,
            "Model": name,
            "NumBands": len(band_indices),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Accuracy": acc,
            "MCC": mcc
        })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")  # 時間戳記
    pd.DataFrame(random_results).to_csv(f"all_result/{concentration}_{bandset_name}_model_random_split_result{ts}.csv", index=False)

    # ---------- Cross Validation (5-fold Stratified) ----------
    # 先把「全部資料」用同樣的 band_indices 載入
    X_all, y_all, _ = load_raw_spectra(vi_subset_df, healthy_dir.parent, label_encoder, band_indices=band_indices)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for name, model in models.items():
        acc_list, sen_list, spe_list, mcc_list = [], [], [], []

        for tr_idx, te_idx in skf.split(X_all, y_all):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            cm = confusion_matrix(y_te, y_pred, labels=np.unique(y_all))
            mcc = matthews_corrcoef(y_te, y_pred)

            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sen = tp / (tp + fn) if (tp + fn) else 0.0
                spe = tn / (tn + fp) if (tn + fp) else 0.0
            else:
                sen = spe = np.nan

            acc_list.append(acc)
            sen_list.append(sen)
            spe_list.append(spe)
            mcc_list.append(mcc)

        print(f"[{bandset_name} | CV] {name}: SEN={np.mean(sen_list):.4f}, SPE={np.mean(spe_list):.4f}, ACC={np.mean(acc_list):.4f}, MCC={np.mean(mcc_list):.4f}")
        cv_results.append({
            "BandSet": bandset_name,
            "Model": name,
            "NumBands": len(band_indices),
            "Sensitivity": float(np.mean(sen_list)),
            "Specificity": float(np.mean(spe_list)),
            "Accuracy": float(np.mean(acc_list)),
            "MCC": float(np.mean(mcc_list))
        })

    ts = datetime.now().strftime("%Y%m%d-%H%M%S") # 時間戳記
    pd.DataFrame(cv_results).to_csv(f"all_result/{concentration}_{bandset_name}_model_crossval_result_{ts}.csv", index=False)

print("\n所有 BandSet 的 Random 與 CV 結果、以及波長對照表，已輸出在 all_result/")

# === Band set: ALL ===
#  target 673.59 nm -> idx 136 (actual 673.59 nm, Δ=0.00 nm)
#  target 655.29 nm -> idx 127 (actual 655.29 nm, Δ=0.00 nm)
#  target 641.02 nm -> idx 120 (actual 641.02 nm, Δ=0.00 nm)
#  target 647.14 nm -> idx 123 (actual 647.14 nm, Δ=0.00 nm)
#  target 618.52 nm -> idx 109 (actual 618.52 nm, Δ=0.00 nm)
#  target 754.36 nm -> idx 176 (actual 754.36 nm, Δ=0.00 nm)
#  target 709.01 nm -> idx 153 (actual 708.01 nm, Δ=1.00 nm)
#  target 681.70 nm -> idx 140 (actual 681.70 nm, Δ=0.00 nm)
#  target 747.31 nm -> idx 172 (actual 746.32 nm, Δ=0.99 nm)
#  target 740.27 nm -> idx 169 (actual 740.28 nm, Δ=0.01 nm)
#  target 1001.59 nm -> idx 299 (actual 1001.59 nm, Δ=0.00 nm)
#  target 997.53 nm -> idx 297 (actual 997.53 nm, Δ=0.00 nm)
#  target 991.45 nm -> idx 294 (actual 991.45 nm, Δ=0.00 nm)
#  target 993.48 nm -> idx 295 (actual 993.48 nm, Δ=0.00 nm)
#  target 451.17 nm -> idx 29 (actual 451.17 nm, Δ=0.00 nm)
#  target 405.86 nm -> idx 8 (actual 405.86 nm, Δ=0.00 nm)
#  target 401.50 nm -> idx 6 (actual 401.50 nm, Δ=0.00 nm)
#  target 410.20 nm -> idx 10 (actual 410.20 nm, Δ=0.00 nm)
#  target 403.68 nm -> idx 7 (actual 403.68 nm, Δ=0.00 nm)
#  target 421.04 nm -> idx 15 (actual 421.04 nm, Δ=0.00 nm)
#  target 433.99 nm -> idx 21 (actual 433.99 nm, Δ=0.00 nm)
#  target 442.59 nm -> idx 25 (actual 442.59 nm, Δ=0.00 nm)
#  target 519.01 nm -> idx 61 (actual 519.01 nm, Δ=0.00 nm)
#  target 645.10 nm -> idx 122 (actual 645.10 nm, Δ=0.00 nm)
#  target 710.03 nm -> idx 154 (actual 710.03 nm, Δ=0.00 nm)
# [ALL | Random] SVM (RBF Kernel): SEN=0.2800, SPE=0.9600, ACC=0.6200, MCC=0.3273
# [ALL | Random] Random Forest: SEN=0.6400, SPE=0.7200, ACC=0.6800, MCC=0.3612
# [ALL | Random] k-NN: SEN=0.6400, SPE=0.6000, ACC=0.6200, MCC=0.2402
# [ALL | Random] Logistic Regression: SEN=0.2800, SPE=0.9200, ACC=0.6000, MCC=0.2603
# [ALL | Random] LDA: SEN=0.6800, SPE=0.8400, ACC=0.7600, MCC=0.5268
# [ALL | Random] XGBoost: SEN=0.6400, SPE=0.7200, ACC=0.6800, MCC=0.3612
# [ALL | CV] SVM (RBF Kernel): SEN=0.2800, SPE=0.9400, ACC=0.6100, MCC=0.2878
# [ALL | CV] Random Forest: SEN=0.6700, SPE=0.7200, ACC=0.6950, MCC=0.3928
# [ALL | CV] k-NN: SEN=0.7400, SPE=0.5800, ACC=0.6600, MCC=0.3266
# [ALL | CV] Logistic Regression: SEN=0.3300, SPE=0.7700, ACC=0.5500, MCC=0.1123
# [ALL | CV] LDA: SEN=0.8200, SPE=0.7300, ACC=0.7750, MCC=0.5598
# [ALL | CV] XGBoost: SEN=0.7000, SPE=0.6500, ACC=0.6750, MCC=0.3593