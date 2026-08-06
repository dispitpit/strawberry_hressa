import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from pathlib import Path
import spectral
import os
from skimage import exposure

spectral.settings.envi_support_nonlowercase_params = True

#*************#
### 選擇日期
day_folders = [
    # "0611","0612","0613","0616","0617","0618","0619","0620",
    # "0621","0624","0625","0702","0703"
    # "0704", "0705", "0708", "0709", "0710"
    "0628"
]

### 選擇檔案號碼
start = 1
end = 20
#*************#

# 資料路徑
datapath_root = Path(r'C:\Users\Amanda\PycharmProjects\test\病害高光譜2024June')
# 專案輸出路徑
datapath_project = Path(r'C:\Users\Amanda\PycharmProjects\test\RGB image')

# 決定要用哪幾個波段合成 RGB
band_R = 120
band_G = 74
band_B = 28

# Gamma 值 (調整亮度： <1 變亮, >1 變暗)
gamma_val = 0.3

plt.ioff()
for day in day_folders:
    # 找到對應天數的資料夾
    datapath_day = datapath_root / f"2024{day}"
    # 輸出資料夾
    output_dir = datapath_project / f"D{day}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(start, end + 1):
        # 組出 hdr / raw 檔名
        hdr_file = datapath_day / f"sample{i}_RT.hdr"
        raw_file = datapath_day / f"sample{i}_RT.raw"

        if not hdr_file.exists():
            print(f"[警告] 找不到標頭檔: {hdr_file}")
            continue
        if not raw_file.exists():
            print(f"[警告] 找不到原始數據檔: {raw_file}")
            continue

        # 開啟 ENVI 檔案
        try:
            image = envi.open(str(hdr_file))
        except Exception as e:
            print(f"[錯誤] 無法開啟 ENVI: {hdr_file}, 錯誤訊息: {e}")
            continue

        # 讀取指定波段 (組成 RGB)
        try:
            # 這裡的 band_R, band_G, band_B 從 0 開始
            img = image.read_bands([band_R, band_G, band_B])  # shape: (height, width, 3)
        except Exception as e:
            print(f"[錯誤] 讀取波段失敗: {hdr_file}, 錯誤訊息: {e}")
            continue

        img_min, img_max = img.min(), img.max()
        img_normalized = (img - img_min) / (img_max - img_min + 1e-8)
        gamma_corrected = exposure.adjust_gamma(img_normalized, gamma=gamma_val)

        output_path = output_dir / f"D{day}_S{i}.png"
        plt.imsave(str(output_path), gamma_corrected)
        # print(f"[Success] 已保存增亮後影像: {output_path}")

print("所有處理完成!")