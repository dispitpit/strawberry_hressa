# 把ALL資料集TRAIN的資料丟過來
# 丟到test0415/重製1003/ks_npy_train_subset
# 完成複製：229 個檔案 → ks_npy_train_subset
import pandas as pd
from pathlib import Path
import shutil

# === 路徑設定 ===
CSV_PATH   = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/KS重製/ks_npy_divided_result/ks_split_train_npy.csv")   # 你的清單
SRC_BASE   = Path(rf"D:/Users/Amanda/PycharmProjects/test/test_0322_new.raw_pca(wait)/npy")
DST_BASE   = Path(r"ks_npy_train_subset")

# === 讀清單 ===
df = pd.read_csv(CSV_PATH)
assert {"File", "Label"}.issubset(df.columns), "CSV 需要包含 File, Label 欄位"

# === 建立輸出資料夾 ===
for sub in ["healthy", "unhealthy"]:
    (DST_BASE / sub).mkdir(parents=True, exist_ok=True)

copied, missing = 0, []

# === 複製 ===
for _, row in df.iterrows():
    fname = str(row["File"])
    label = str(row["Label"]).lower().strip()
    src   = SRC_BASE / label / fname
    dst   = DST_BASE / label / fname

    if src.exists():
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing.append(str(src))

# === 摘要 ===
print(f"完成複製：{copied} 個檔案 → {DST_BASE}")
if missing:
    print(f"找不到 {len(missing)} 個檔案，前幾個：")
    for p in missing[:10]:
        print(" -", p)
