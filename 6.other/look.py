# 輸入檔名執行
# 右下special variables 點 a 展開即可查看
from pathlib import Path
import numpy as np

p = Path.cwd() / "wavelengths.npy"
print("CWD =", Path.cwd())
a = np.load(p, mmap_mode='r')
print(a.shape, a.dtype, np.nanmin(a), np.nanmax(a))

# .npy 是numpy專用的單一陣列二進位儲存格式
# *不丟資料
# *二進位直存直讀，快

# 如果要存多陣列要用.npz (加壓縮)