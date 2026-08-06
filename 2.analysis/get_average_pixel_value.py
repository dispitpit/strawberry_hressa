# 1001復盤OK
# 讀一張高光譜影像 → 用 mask 挑出葉片區域 → 算該區域的平均光譜
# 製作npy和csv資料
import numpy as np
import spectral
import cv2
import pandas as pd
from pathlib import Path
spectral.settings.envi_support_nonlowercase_params = True

# === 資料來源路徑 ===
hsi_root = Path(r"C:\Users\Amanda\PycharmProjects\test\病害高光譜2024June")
mask_root = Path(r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask_cleaned")

# === 輸出資料夾（分開）===
output_csv = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\csv")
output_npy = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\npy")
output_csv.mkdir(parents=True, exist_ok=True)
output_npy.mkdir(parents=True, exist_ok=True)

valid_dates = [
    "0607", "0611", "0612", "0613", "0616", "0617",
    "0618", "0619", "0620", "0621", "0624", "0625",
    "0627", "0628", "0702", "0703", "0704", "0705",
    "0708", "0709", "0710"
]

# === 日期迴圈 ===
for date_short  in valid_dates: # D0607 到 D0701
    date_str = "2024" + date_short
    folder = hsi_root / date_str
    if not folder.exists():
        print(f"跳過不存在的資料夾: {folder}")
        continue

    for i in range(1, 21):  # sample1 到 sample20
        sample_name = f"sample{i}_RT"
        # .hdr
        hdr_path = folder / f"{sample_name}.hdr"
        # .raw
        # 嘗試偵測 .raw 路徑（無副檔名 or .raw 副檔名）
        raw_base = folder / sample_name
        raw_path = raw_base if raw_base.exists() else raw_base.with_suffix(".raw")

        # 對應 mask 名稱
        # date_str = f"{day:04d}"  # e.g., 0620
        mask_name = f"D{date_short}_S{i}_mask.png"
        mask_path = mask_root / mask_name

        # 檢查檔案是否齊全
        if not (hdr_path.exists() and raw_path.exists() and mask_path.exists()):
            print(f"缺少檔案，跳過：D{date_short}_S{i}")
            continue

        # 讀取影像和 mask
        try:
            # ***用envi去讀取***
            # img => 讀兩個大寶
            # cube => (H,W,B), 儲存numpy形式
            img = spectral.envi.open(str(hdr_path), str(raw_path))
            cube_np = np.array(img.load())
        except Exception as e:
            print(f"無法讀取影像：{sample_name} — {e}")
            continue

        # 讀準備好的mask，並檢查尺寸是否一致
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != cube_np.shape[:2]:
            print(f"Mask 錯誤或尺寸不符：{mask_name}")
            continue

        # 做ROI
        roi_mask = mask > 0
        leaf_pixels = cube_np[roi_mask]
        if leaf_pixels.shape[0] == 0:
            print(f"ROI 區域為空：D{date_short}_S{i}")
            continue

        # 做完之後將ROI取平均值 => 變成一條光譜線
        mean_spectrum = np.mean(leaf_pixels, axis=0)

        # 儲存結果
        filename = f"D{date_short}_S{i}_mean_spectrum"
        # print(f"儲存成功: {filename}")

        pd.DataFrame(mean_spectrum, columns=["Reflectance"]).to_csv(output_csv / f"{filename}.csv", index_label="Band")
        np.save(output_npy / f"{filename}.npy", mean_spectrum)

        print(f"儲存完成：{filename}")
