# 1001復盤ok
# 將分類成感染與未感染的資料夾再分成高濃度與低濃度
# ****不可以重跑****
# low跟high有再整理過!!!
import shutil
from pathlib import Path
import re

# 原始資料夾
npy_root = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\npy")
healthy_src = npy_root / "healthy"
unhealthy_src = npy_root / "unhealthy"

# 目標資料夾
low_healthy_dst = npy_root.parent / "low" / "healthy"
low_unhealthy_dst = npy_root.parent / "low" / "unhealthy"
high_healthy_dst = npy_root.parent / "high" / "healthy"
high_unhealthy_dst = npy_root.parent / "high" / "unhealthy"

# 建立資料夾
for folder in [low_healthy_dst, low_unhealthy_dst, high_healthy_dst, high_unhealthy_dst]:
    folder.mkdir(parents=True, exist_ok=True)

# 分類函數
def classify_and_copy(source_dir, low_dst, high_dst):
    for file in source_dir.glob("*.npy"):
        # 取出盆栽標籤
        match = re.search(r"_S(\d+)", file.stem)
        if not match:
            continue
        sample_num = int(match.group(1))
        if sample_num <= 5 or sample_num >= 16:
            shutil.copy2(file, low_dst / file.name)
        if sample_num <= 15:
            shutil.copy2(file, high_dst / file.name)

# 執行分類與複製
# 分別去檢查兩個npy裡面的資料夾去丟
classify_and_copy(healthy_src, low_healthy_dst, high_healthy_dst)
classify_and_copy(unhealthy_src, low_unhealthy_dst, high_unhealthy_dst)

print("finish")