import numpy as np
import spectral
import cv2
import pandas as pd
from pathlib import Path

spectral.settings.envi_support_nonlowercase_params = True

# === 資料來源 ===
hsi_root = Path(r"C:\Users\Amanda\PycharmProjects\test\病害高光譜2024June")
mask_root = Path(r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask_cleaned")

# === 輸出 ===
base_csv = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0607\csv")
base_npy = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0607\npy")


stat_methods = ["mean", "median", "std", "max", "min"]
# 建立子資料夾
csv_dirs = {stat: (base_csv / stat) for stat in stat_methods}
npy_dirs = {stat: (base_npy / stat) for stat in stat_methods}
for dir_path in list(csv_dirs.values()) + list(npy_dirs.values()):
    dir_path.mkdir(parents=True, exist_ok=True)

# === 有效日期 ===
valid_dates = [
    "0607", "0611", "0612", "0613", "0616", "0617",
    "0618", "0619", "0620", "0621", "0624", "0625",
    "0627", "0628", "0702", "0703", "0704", "0705",
    "0708", "0709", "0710"
]

# === 處理資料 ===
for date_short in valid_dates:
    date_str = "2024" + date_short
    folder = hsi_root / date_str
    if not folder.exists():
        print(f"跳過不存在的資料夾: {folder}")
        continue

    for i in range(1, 21):
        sample_name = f"sample{i}_RT"
        hdr_path = folder / f"{sample_name}.hdr"
        raw_base = folder / sample_name
        raw_path = raw_base if raw_base.exists() else raw_base.with_suffix(".raw")

        mask_name = f"D{date_short}_S{i}_mask.png"
        mask_path = mask_root / mask_name

        if not (hdr_path.exists() and raw_path.exists() and mask_path.exists()):
            print(f"缺少檔案，跳過：D{date_short}_S{i}")
            continue

        try:
            img = spectral.envi.open(str(hdr_path), str(raw_path))
            cube_np = np.array(img.load())
        except Exception as e:
            print(f"無法讀取影像：{sample_name} — {e}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != cube_np.shape[:2]:
            print(f"Mask 錯誤或尺寸不符：{mask_name}")
            continue

        roi_mask = mask > 0
        leaf_pixels = cube_np[roi_mask]
        if leaf_pixels.shape[0] == 0:
            print(f"ROI 區域為空：D{date_short}_S{i}")
            continue

        stats = {
            "mean": np.mean(leaf_pixels, axis=0),
            "median": np.median(leaf_pixels, axis=0),
            "std": np.std(leaf_pixels, axis=0),
            "max": np.max(leaf_pixels, axis=0),
            "min": np.min(leaf_pixels, axis=0)
        }

        for stat_name, spectrum in stats.items():
            filename = f"D{date_short}_S{i}_spectrum"
            np.save(npy_dirs[stat_name] / f"{filename}.npy", spectrum)
            pd.DataFrame(spectrum, columns=["Reflectance"]).to_csv(
                csv_dirs[stat_name] / f"{filename}.csv", index_label="Band"
            )

        print(f"儲存完成：D{date_short}_S{i}")
