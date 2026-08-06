# 0906  update
# 低濃度
# [✓] train.txt → 84 samples, 25200 png 路徑（D:\Users\Amanda\PycharmProjects\test\test_0305_bands300_mask\ks_split_lists\low\train.txt）
# [✓] val.txt → 20 samples, 6000 png 路徑（D:\Users\Amanda\PycharmProjects\test\test_0305_bands300_mask\ks_split_lists\low\val.txt）
# [✓] test.txt → 36 samples, 10800 png 路徑（D:\Users\Amanda\PycharmProjects\test\test_0305_bands300_mask\ks_split_lists\low\test.txt）
# [KS split 內容檢查]
# train -> healthy=12600, diseased=12600
# val   -> healthy=3000, diseased=3000
# test  -> healthy=5400, diseased=5400

# 高濃度
# [✓] train.txt → 120 samples, 36000 png 路徑（D:\Users\Amanda\PycharmProjects\test\test_0305_bands300_mask\ks_split_lists\high\train.txt）
# [✓] val.txt → 30 samples, 9000 png 路徑（D:\Users\Amanda\PycharmProjects\test\test_0305_bands300_mask\ks_split_lists\high\val.txt）
# [✓] test.txt → 50 samples, 15000 png 路徑（D:\Users\Amanda\PycharmProjects\test\test_0305_bands300_mask\ks_split_lists\high\test.txt）
# [KS split 內容檢查]
# train -> healthy=18000, diseased=18000
# val   -> healthy=4500, diseased=4500
# test  -> healthy=7500, diseased=7500

# ------------------------------------------------------------

# generate_ks_lists.py
# ------------------------------------------------------------
# 先安裝依賴：pip install numpy pandas scikit-learn tqdm
# 執行：python generate_ks_lists.py  (--concentration low) (--depth 300)

# # 低濃度
# python generate_ks_lists.py --concentration low --depth 300
# result:
# (.venv) PS C:\Users\Amanda\PycharmProjects\test\test_0801_DL> python generate_ks_lists.py --concentration low --depth 300
# [✓] train.txt → 84 samples （C:\Users\Amanda\PycharmProjects\test\test_0801_DL\ks_split_lists\low\train.txt）
# [✓] val.txt → 20 samples （C:\Users\Amanda\PycharmProjects\test\test_0801_DL\ks_split_lists\low\val.txt）
# [✓] test.txt → 36 samples （C:\Users\Amanda\PycharmProjects\test\test_0801_DL\ks_split_lists\low\test.txt）

# # 高濃度
# python generate_ks_lists.py --concentration high --depth 300
# result:
# (.venv) PS C:\Users\Amanda\PycharmProjects\test\test_0801_DL> python generate_ks_lists.py --concentration high --depth 300
# [✓] train.txt → 120 samples （C:\Users\Amanda\PycharmProjects\test\test_0801_DL\ks_split_lists\high\train.txt）
# [✓] val.txt → 30 samples （C:\Users\Amanda\PycharmProjects\test\test_0801_DL\ks_split_lists\high\val.txt）
# [✓] test.txt → 50 samples （C:\Users\Amanda\PycharmProjects\test\test_0801_DL\ks_split_lists\high\test.txt）

# output
# test_0305_bands300_mask/
# └─ ks_split_lists/
#    ├─ low/
#    │  ├─ train.txt
#    │  ├─ val.txt
#    │  └─ test.txt
#    └─ high/
#       ├─ train.txt
#       ├─ val.txt
#       └─ test.txt

# generate_ks_lists.py
# ------------------------------------------------------------
# 用法：
#   python generate_ks_lists.py --concentration low  --depth 300
#   python generate_ks_lists.py --concentration high --depth 300
# 會輸出：
#   <BAND_ROOT>/ks_split_lists/<conc>/{train,val,test}.txt
# ------------------------------------------------------------


import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
# ---------------------- 可自行調整 ----------------------
PROJ_ROOT  = Path(r"D:/Users/Amanda/PycharmProjects/test")
VI_CSV     = PROJ_ROOT / "test_0322_new.raw_pca(wait)" / "vi_40_output.csv"

# 300-band png 在 test_0305_bands300_mask
BAND_ROOT  = PROJ_ROOT / "test_0801_DL"

VI_FEATURES = ["MTCI", "VOG3", "VOG1", "Chlorophyll Index RedEdge", "CInir"]
TRAIN_RATIO, VAL_RATIO = 0.60, 0.15       # test = 1 - sum

# train = 3/4 val = 1/5
# --------------------------------------------------------
def kennard_stone(X, n_train):
    """簡版 KS：用歐氏距離挑最分散的 n_train 個樣本。"""
    D = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [i, j]
    while len(selected) < n_train:
        rest = list(set(range(len(X))) - set(selected))
        min_dist = [min(D[r][selected]) for r in rest]
        selected.append(rest[int(np.argmax(min_dist))])
    return np.array(selected, dtype=int)

def sample_to_pngs(b_root: Path, conc: str, label: str, sample_id: str, depth: int):
    """
    用 sample_id（如 D0618_S16）找到該樣本的 300 張波段 png。
    CSV 的 Label 可能是 'unhealthy'，而影像資料夾是 'diseased'，這裡做一次對應。
    """
    label = label.lower()
    folder_label = "diseased" if label in {"unhealthy", "diseased"} else "healthy"
    pattern = f"{sample_id}_B*_masked.png"
    png_dir = b_root / f"{conc}_concentration" / folder_label
    return sorted(png_dir.glob(pattern))[:depth]

def count_labels_in_txt(txt_path: Path) -> Counter:
    c = Counter()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().lower()
            if not p:
                continue
            if "healthy" in p:
                c["healthy"] += 1
            elif ("diseased" in p) or ("unhealthy" in p):
                c["diseased"] += 1
    return c

def main(conc: str, depth: int):
    band_root  = BAND_ROOT
    split_root = band_root / "ks_split_lists" / conc
    split_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(VI_CSV)
    df["Label"] = df["Label"].str.lower()

    raw_root = PROJ_ROOT / "test_0322_new.raw_pca(wait)" / conc
    healthy_npy_dir   = raw_root / "healthy"
    unhealthy_npy_dir = raw_root / "unhealthy"
    selected_files = {p.name for p in healthy_npy_dir.glob("*.npy")} | \
                     {p.name for p in unhealthy_npy_dir.glob("*.npy")}

    df = df[df["File"].isin(selected_files)].reset_index(drop=True)

    file_to_label = dict(zip(df["File"], df["Label"]))

    # ----------  針對 healthy / unhealthy 各跑 KS ----------
    lists = {"train": [], "val": [], "test": []}

    for cls in ["healthy", "unhealthy"]:
        sub = df[df["Label"] == cls].reset_index(drop=True)

        if len(sub) == 0:
            print(f"[Warn] {conc}: 類別 {cls} 沒有樣本，跳過。")
            continue

        if len(sub) < 3:
            print(f"[Warn] {conc}: 類別 {cls} 只有 {len(sub)} 個樣本，無法 KS；全部先放進 train。")
            lists["train"].extend(sub["File"].tolist())
            continue

        X = sub[VI_FEATURES].values.astype(float)
        n = len(sub)
        n_train = max(1, int(TRAIN_RATIO * n))
        n_val   = max(0, int(VAL_RATIO   * n))

        ks_idx  = kennard_stone(X, n_train)
        rest    = list(set(range(n)) - set(ks_idx))
        val_idx = rest[:n_val]
        test_idx= rest[n_val:]

        split_map = {"train": ks_idx, "val": val_idx, "test": test_idx}
        for tag, idxs in split_map.items():
            lists[tag].extend(sub.loc[idxs, "File"].tolist())

    for tag, file_names in lists.items():
        out_txt = split_root / f"{tag}.txt"
        cnt_lines = 0
        with open(out_txt, "w", encoding="utf-8") as f:
            for fname in file_names:
                label = file_to_label.get(fname, "healthy")
                sample_id = fname.split("_mean")[0]   # 例：D0618_S16
                pngs = sample_to_pngs(band_root, conc, label, sample_id, depth)
                if not pngs:
                    print(f"[WARN] 找不到 png：{conc}/{label}/{sample_id}（檢查路徑與命名）")
                for p in pngs:
                    f.write(str(p) + "\n")
                    cnt_lines += 1
        print(f"[✓] {tag}.txt → {len(file_names)} samples, {cnt_lines} png 路徑（{out_txt}）")

    print("\n[KS split 內容檢查]")
    for tag in ("train", "val", "test"):
        txt = split_root / f"{tag}.txt"
        c = count_labels_in_txt(txt)
        print(f"{tag:5s} -> healthy={c.get('healthy',0)}, diseased={c.get('diseased',0)}")
        if min(c.get("healthy",0), c.get("diseased",0)) == 0:
            print(f"[WARN] {tag} 裡某一類為 0，請調整 KS 或手動移動少量樣本避免單一類別。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concentration", default="low", choices=["low", "high"])
    parser.add_argument("--depth", type=int, default=300, help="每組多少 png 組成一 cube")
    args = parser.parse_args()
    main(args.concentration, args.depth)