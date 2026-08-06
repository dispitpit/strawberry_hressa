import os
import re
import shutil
from pathlib import Path

train_set_folder = Path(r"C:\Users\Amanda\PycharmProjects\test\RGB image\count_mIOU\train_set")
u2net_final_folder = Path(r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask")
output_folder = Path(r"C:\Users\Amanda\PycharmProjects\test\RGB image\count_mIOU\u2net_mask_result")
output_folder.mkdir(parents=True, exist_ok=True)

def parse_day_s(filename):
    """
    假設檔名類似: D0703_S20_leaf_mask.png
    解析出 (day=703, sample=20)
    若失敗則回傳 None
    """
    # 移除副檔名
    base, ext = os.path.splitext(filename)
    # 用正則擷取: ^(D\d+)_S(\d+)
    # e.g. D0703_S20 => day_str='D0703', sample_str='20'
    m = re.match(r'^(D\d+)_S(\d+)', base)
    if not m:
        return None
    day_str = m.group(1)  # e.g. 'D0703'
    sample_str = m.group(2)  # e.g. '20'
    try:
        day_num = int(day_str[1:])    # 去除 'D'
        sample_num = int(sample_str)  # 20
        return (day_num, sample_num)
    except:
        return None

# 1. 從 train_set 解析 day, sample, 放到 set
train_set_tuples = set()
for f in train_set_folder.glob("*.png"):
    day_s = parse_day_s(f.name)
    if day_s:
        train_set_tuples.add(day_s)
    else:
        print(f"[警告] {f.name} 無法解析 day, sample")

# 2. 在 u2net_final_folder 中，比對 day, sample
count_copied = 0
for f in u2net_final_folder.glob("*.png"):
    day_s = parse_day_s(f.name)
    if day_s and day_s in train_set_tuples:
        # 複製
        dst_path = output_folder / f.name
        shutil.copy2(str(f), str(dst_path))
        count_copied += 1
        print(f"複製 {f.name} -> {dst_path}")

print(f"Finish! 總共複製 {count_copied} 個檔案到 {output_folder}")
