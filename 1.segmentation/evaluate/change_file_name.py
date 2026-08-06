import re
import os
from pathlib import Path

# 要處理的資料夾路徑
folder = Path(r"C:\Users\Amanda\PycharmProjects\test\RGB image\count_mIOU\u2net_mask_result")

# 迭代該資料夾中的所有檔案 (這裡假設都是 .png，如需支援更多副檔名可自行擴充)
for file_path in folder.glob("*.*"):
    if file_path.is_dir():
        continue  # 跳過資料夾

    file_name = file_path.name  # e.g. "D0617_S11_leaf_mask_something.png"
    base, ext = os.path.splitext(file_name)  # e.g. base="D0617_S11_leaf_mask_something", ext=".png"

    # 用正則擷取前綴 "Dxxxx_Sxx"
    # 例如 D0617_S11 => group(1)='D0617_S11'
    # 若檔名有更多複雜字樣，請依需求調整
    m = re.match(r"^(D\d+_S\d+)", base)
    if not m:
        print(f"[警告] {file_name} 無法解析前綴，跳過")
        continue

    prefix = m.group(1)  # e.g. "D0617_S11"

    # 新檔名 (保留副檔名，這裡改成 prefix + ".png")
    # 如果想完全不保留原副檔名，就直接用 ".png" 或你要的副檔
    new_name = prefix + ".png"

    # 若要確保不覆蓋原檔，可檢查新檔名是否已存在
    new_path = folder / new_name
    if new_path.exists():
        print(f"[警告] {new_name} 已存在，跳過以免覆蓋")
        continue

    # 執行重新命名
    file_path.rename(new_path)
    print(f"重命名 {file_name} -> {new_name}")
