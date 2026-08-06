# 1001復盤ok
# 將資料集分成感染與未感染
# ---> npy / healthy or unhealthy
# 可改,npy或,csv
from pathlib import Path
import shutil
import re
import pandas as pd
# import ace_tools as tools; tools.display_dataframe_to_user(name="Healthy/Unhealthy File Summary", dataframe=pd.DataFrame(data))

# 來源資料夾
source_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\csv")

# 分類輸出資料夾
healthy_dir = source_dir / "healthy"
unhealthy_dir = source_dir / "unhealthy"
healthy_dir.mkdir(parents=True, exist_ok=True)
unhealthy_dir.mkdir(parents=True, exist_ok=True)

# 正則表達式抓日期與樣本編號
# \d: 組成群組 4代表4位數 +代表一位數以上
# ex:                     0628    15
pattern = re.compile(r"D(\d{4})_S(\d+)_mean_spectrum")

# 處理所有 .npy 檔案
for file in source_dir.glob("*.csv"):
    match = pattern.search(file.stem)
    if not match:
        continue

    # 用match.groups()取前面pattern取出的group值
    # ex:
    # y = m.group(1)   # '2023'  ← 第1群組
    # s = m.group(2)   # '15'    ← 第2群組
    date_str, sample_str = match.groups()
    # 將取出的日期和第幾筆轉成數值
    # 轉成數字就不會有前字串11識別1為true的情況發生
    date = int(date_str)
    sample = int(sample_str)

    # 分類規則
    # 日期小於0617一律為未感染
    # 日期大於0617之後檢查編號，編號0-5為未感染，其他為感染
    if date <= 617:
        target_dir = healthy_dir
    else:
        target_dir = healthy_dir if 1 <= sample <= 5 else unhealthy_dir

    # 複製檔案
    shutil.copy2(file, target_dir / file.name)

# 統計檔案
healthy_files = sorted(f.name for f in healthy_dir.glob("*.csv"))
unhealthy_files = sorted(f.name for f in unhealthy_dir.glob("*.csv"))

# 對齊顯示
max_len = max(len(healthy_files), len(unhealthy_files))
data = {
    "Healthy": healthy_files + [""] * (max_len - len(healthy_files)),
    "Unhealthy": unhealthy_files + [""] * (max_len - len(unhealthy_files))
}

