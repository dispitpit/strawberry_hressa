# utils_records.py
# 作用：
# 1. 讀取真實資料的 split CSV，整理成統一 records DataFrame
# 2. 掃描 CycleGAN fake 資料夾，整理成 augmentation records DataFrame

from pathlib import Path
import pandas as pd
import numpy as np


def _pick_col(df, candidates, required=True, col_desc="欄位"):
    """從候選欄位名中找出實際存在的欄位。"""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"找不到{col_desc}。目前 CSV 欄位有：{list(df.columns)}\n"
            f"候選名稱嘗試過：{candidates}"
        )
    return None


def _normalize_label(val):
    """把 label 統一轉成 0 / 1。"""
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return int(val)

    s = str(val).strip().lower()
    if s in ["healthy", "health", "normal", "a", "class0", "0", "0.0"]:
        return 0
    if s in ["unhealthy", "disease", "diseased", "infected", "b", "class1", "1", "1.0"]:
        return 1

    try:
        return int(float(s))
    except Exception as e:
        raise ValueError(f"無法辨識 label 值：{val}") from e


def _label_to_folder(label_int: int) -> str:
    return "healthy" if int(label_int) == 0 else "unhealthy"


def _canonical_stem(name_or_stem: str) -> str:
    """
    把檔名主體統一成真正 sample id。

    例：
    - D0611_S1.npy -> D0611_S1
    - D0611_S1_mean_spectrum.npy -> D0611_S1
    - D0611_S1_mean_spectrum_fakeB.npy -> D0611_S1
    - D0611_S1_fake.npy -> D0611_S1
    """
    stem = Path(name_or_stem).stem
    changed = True
    while changed:
        changed = False
        for suf in ["_fakeA", "_fakeB", "_fake", "_mean_spectrum"]:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
    return stem


def _candidate_real_names(raw_file: str):
    """
    根據 CSV 裡的名稱，產生對應的 2D CNN cube 檔名。
    例如 CSV 若寫 D0611_S1_mean_spectrum.npy，這裡會同時嘗試：
    - D0611_S1_mean_spectrum.npy
    - D0611_S1.npy
    """
    p = Path(str(raw_file).strip())
    names = [p.name]

    canonical = _canonical_stem(p.name) + ".npy"
    if canonical not in names:
        names.append(canonical)

    return names


def _resolve_real_path(row, base_data_root: Path, file_col, label_col=None):
    """
    把 CSV 的 File 欄位對應到真正的 2D cube 路徑。
    """
    raw_file = str(row[file_col]).strip()
    p = Path(raw_file)
    candidate_names = _candidate_real_names(raw_file)

    tried = []

    # 1) 若 CSV 已給絕對路徑，先照原始名稱與轉換後名稱都試一次
    if p.is_absolute():
        for name in candidate_names:
            cand = p.with_name(name)
            tried.append(str(cand))
            if cand.exists():
                return cand

    # 2) 若 CSV 給相對路徑，試 base_data_root / 原路徑(但替換可能的檔名)
    parent_rel = p.parent if str(p.parent) != "." else Path("")
    for name in candidate_names:
        cand = base_data_root / parent_rel / name
        tried.append(str(cand))
        if cand.exists():
            return cand

    # 3) 若有 label，依 label 對應到 healthy/unhealthy 子資料夾再試
    if label_col is not None:
        folder = _label_to_folder(_normalize_label(row[label_col]))
        for name in candidate_names:
            cand = base_data_root / folder / name
            tried.append(str(cand))
            if cand.exists():
                return cand

    # 4) 保守起見，healthy / unhealthy 都試一次
    for folder in ["healthy", "unhealthy"]:
        for name in candidate_names:
            cand = base_data_root / folder / name
            tried.append(str(cand))
            if cand.exists():
                return cand

    raise FileNotFoundError(
        f"找不到真實資料檔案：{raw_file}\n"
        f"可能原因：CSV 內仍是 mean_spectrum 檔名，但 2D CNN 資料夾存的是 cube 檔名。\n"
        f"已嘗試路徑：\n- " + "\n- ".join(tried)
    )


def build_real_records(csv_path, base_data_root):
    """
    讀取 train/test split CSV，輸出統一 records：
    path / label / origin_id / is_fake
    """
    df = pd.read_csv(csv_path)

    print("[INFO] build_real_records()")
    print("[INFO] CSV path:", csv_path)
    print("[INFO] CSV columns:", list(df.columns))
    print("[INFO] CSV size:", df.shape)

    file_col = _pick_col(
        df,
        candidates=["path", "Path", "filepath", "file_path", "filename", "Filename", "file", "File"],
        required=True,
        col_desc="檔名/路徑欄位"
    )
    label_col = _pick_col(
        df,
        candidates=["label", "Label", "class", "Class", "class_name", "ClassName", "target", "Target", "y", "Y"],
        required=True,
        col_desc="標籤欄位"
    )
    sample_id_col = _pick_col(
        df,
        candidates=["sample_id", "SampleID", "sampleID", "id", "ID", "origin_id", "OriginID"],
        required=False,
        col_desc="sample id 欄位"
    )

    rows = []
    for _, row in df.iterrows():
        resolved_path = _resolve_real_path(
            row=row,
            base_data_root=Path(base_data_root),
            file_col=file_col,
            label_col=label_col,
        )

        label_int = _normalize_label(row[label_col])
        if sample_id_col is not None:
            origin_id = _canonical_stem(str(row[sample_id_col]).strip())
        else:
            origin_id = _canonical_stem(resolved_path.name)

        rows.append({
            "path": str(resolved_path),
            "label": int(label_int),
            "origin_id": str(origin_id),
            "is_fake": 0,
        })

    out_df = pd.DataFrame(rows, columns=["path", "label", "origin_id", "is_fake"])
    print("[INFO] Real records built:", len(out_df))
    print("[INFO] label counts:\n", out_df["label"].value_counts(dropna=False))
    return out_df


def build_aug_records_from_folders(
    aug_root,
    trainA_fakeB_dir="trainA_fakeB",
    trainB_fakeA_dir="trainB_fakeA",
    a_label=0,
    b_label=1,
):
    """
    掃描 fake augmentation 資料夾，輸出：
    path / label / origin_id / is_fake

    預設 A=healthy, B=unhealthy：
    - trainA_fakeB: A -> fakeB，所以 label 應是 B
    - trainB_fakeA: B -> fakeA，所以 label 應是 A
    """
    aug_root = Path(aug_root)
    rows = []

    print("[INFO] build_aug_records_from_folders()")
    print("[INFO] AUG root:", aug_root)

    dir_a2b = aug_root / trainA_fakeB_dir
    if dir_a2b.exists():
        files = sorted(dir_a2b.glob("*.npy"))
        print(f"[INFO] {dir_a2b} -> {len(files)} npy files")
        for p in files:
            rows.append({
                "path": str(p),
                "label": int(b_label),
                "origin_id": _canonical_stem(p.name),
                "is_fake": 1,
            })
    else:
        print(f"[WARN] augmentation folder not found: {dir_a2b}")

    dir_b2a = aug_root / trainB_fakeA_dir
    if dir_b2a.exists():
        files = sorted(dir_b2a.glob("*.npy"))
        print(f"[INFO] {dir_b2a} -> {len(files)} npy files")
        for p in files:
            rows.append({
                "path": str(p),
                "label": int(a_label),
                "origin_id": _canonical_stem(p.name),
                "is_fake": 1,
            })
    else:
        print(f"[WARN] augmentation folder not found: {dir_b2a}")

    out_df = pd.DataFrame(rows, columns=["path", "label", "origin_id", "is_fake"])
    print("[INFO] Aug records built:", len(out_df))
    if len(out_df) > 0:
        print("[INFO] aug label counts:\n", out_df["label"].value_counts(dropna=False))
    return out_df
