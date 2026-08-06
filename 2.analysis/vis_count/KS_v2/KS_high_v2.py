'''
=== 輸出完成 ===
Train list : ks_high_divided_result\ks_split_train_high.csv
Test  list : ks_high_divided_result\ks_split_test_high.csv
Summary    : ks_high_divided_result\ks_split_summary_high.csv
'''

'''分割摘要：
           train  test
Label
Healthy       75    25
Unhealthy     75    25
TOTAL        150    50
'''

# -*- coding: utf-8 -*-
"""
KS split per label (3:1) -> save train/test lists and summary.
No modeling.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ====================== 使用者設定 ======================
CONCENTRATION = "high"  # 對應 {CONCENTRATION}/healthy, {CONCENTRATION}/unhealthy
VI_CSV_PATH = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/new_vis1105/KS_v2/vi_40_high_v2.csv")
BASE_DIR = Path(fr"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/{CONCENTRATION}")

# 選定 VI 欄位
VI_FEATURES = ["VOG1", "NDRE", "MTCI", "RENDVI", "MCARI"]
# 每個 Label 內部訓練比例（3:1 = 0.75）
TRAIN_RATIO = 0.75

OUT_DIR = Path(f"ks_{CONCENTRATION}_divided_result")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ====================== KS 函式 ======================
def kennard_stone(X: np.ndarray, n_train: int) -> np.ndarray:
    """Kennard–Stone (maximin) 以歐氏距離挑代表性訓練集索引。"""
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.arange(n_samples)
    if n_train >= n_samples:
        n_train = n_samples - 1

    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(i), int(j)]

    while len(selected) < n_train:
        remaining = list(set(range(n_samples)) - set(selected))
        min_d = [float(np.min([dist[r, s] for s in selected])) for r in remaining]
        next_idx = remaining[int(np.argmax(min_d))]
        selected.append(next_idx)

    return np.array(selected, dtype=int)

def ks_split_by_label(df: pd.DataFrame, feature_cols, train_ratio=0.75):
    """對每個 Label 單獨做 KS，依 train_ratio 切分，再合併回整體。"""
    trains, tests = [], []
    for label, sub in df.groupby("Label", sort=False):
        # X = sub[feature_cols].to_numpy()
        # 對每個vis做正規化
        X = StandardScaler().fit_transform(sub[feature_cols].to_numpy())
        n = len(sub)
        if n < 3:
            warnings.warn(f"[{label}] 樣本過少 (n={n})，可能無法 3:1 分割。將盡量保留。")
        n_train = min(max(2, int(round(train_ratio * n))), n - 1) if n >= 3 else max(1, n - 1)
        idx_tr = kennard_stone(X, n_train=n_train)
        sub_tr = sub.iloc[idx_tr]
        sub_te = sub.drop(sub_tr.index)
        trains.append(sub_tr)
        tests.append(sub_te)

        # print(f"[{label}] n={n}, train={len(sub_tr)}, test={len(sub_te)}")
    train_df = pd.concat(trains, axis=0).reset_index(drop=True)
    test_df = pd.concat(tests, axis=0).reset_index(drop=True)
    return train_df, test_df

# ====================== 主流程 ======================
def main():
    healthy_dir = BASE_DIR / "healthy"
    unhealthy_dir = BASE_DIR / "unhealthy"
    assert VI_CSV_PATH.exists(), f"VI CSV 不存在：{VI_CSV_PATH}"
    assert healthy_dir.exists() and unhealthy_dir.exists(), f"找不到 {healthy_dir} 或 {unhealthy_dir}"

    vi_df = pd.read_csv(VI_CSV_PATH)
    need_cols = {"File", "Label"} | set(VI_FEATURES)
    missing_cols = need_cols - set(vi_df.columns)
    if missing_cols:
        raise ValueError(f"VI CSV 缺少必要欄位：{missing_cols}")

    existing_files = {f.name for f in healthy_dir.glob("*.npy")} | {f.name for f in unhealthy_dir.glob("*.npy")}
    vi_subset = vi_df[vi_df["File"].isin(existing_files)].copy()
    if vi_subset.empty:
        raise ValueError("篩選後沒有任何檔案匹配到當前濃度資料夾。")

    # KS 分割
    train_df, test_df = ks_split_by_label(vi_subset, VI_FEATURES, train_ratio=TRAIN_RATIO)

    # 輸出清單
    train_list_csv = OUT_DIR / f"ks_split_train_{CONCENTRATION}.csv"
    test_list_csv = OUT_DIR / f"ks_split_test_{CONCENTRATION}.csv"
    train_df[["File", "Label"]].to_csv(train_list_csv, index=False)
    test_df[["File", "Label"]].to_csv(test_list_csv, index=False)

    # 摘要
    def summarize(df, name):
        return df.groupby("Label").size().rename(name)

    summary = pd.concat([summarize(train_df, "train"), summarize(test_df, "test")], axis=1).fillna(0).astype(int)
    summary.loc["TOTAL"] = summary.sum()
    summary_csv = OUT_DIR / f"ks_split_summary_{CONCENTRATION}.csv"
    summary.to_csv(summary_csv)

    # 額外存一份 meta（方便追蹤）
    meta = {
        "concentration": CONCENTRATION,
        "train_ratio": TRAIN_RATIO,
        "vi_features": "|".join(VI_FEATURES),
        "n_train_total": int(len(train_df)),
        "n_test_total": int(len(test_df)),
    }
    pd.DataFrame([meta]).to_csv(OUT_DIR / f"ks_split_meta_{CONCENTRATION}.csv", index=False)

    print("\n=== 輸出完成 ===")
    print(f"Train list : {train_list_csv}")
    print(f"Test  list : {test_list_csv}")
    print(f"Summary    : {summary_csv}")
    # print("\n分割摘要：")
    # print(summary)

if __name__ == "__main__":
    main()
