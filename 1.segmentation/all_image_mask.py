# 建立MASK資料分類
import cv2
import os
import re
import glob


MASK_FOLDER = r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask_cleaned"
ALL_IMAGE_FOLDER = r"C:\Users\Amanda\PycharmProjects\test\all_image"
OUTPUT_FOLDER = r"C:\Users\Amanda\PycharmProjects\test\all_image_mask"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def parse_mask_filename(filename):
    """
    假設檔名為 D0611_S1_mask.png
    - 先去除副檔名 -> D0611_S1_mask
    - 用正則匹配: ^(D\d+_S\d+)_mask$
      例: D0611_S1_mask -> prefix='D0611_S1'
    - 進一步分離 day = 'D0611'
    """
    base, ext = os.path.splitext(filename)  # e.g. base='D0611_S1_mask'
    m = re.match(r'^(D\d+_S\d+)_mask$', base)
    if not m:
        return None

    prefix = m.group(1)
    m2 = re.match(r'^(D\d+)_S(\d+)$', prefix)
    if not m2:
        return None
    day = m2.group(1)  # e.g. 'D0611'
    return day, prefix

# 逐一讀取 mask 資料夾內的檔案
for mask_file in os.listdir(MASK_FOLDER):
    if not mask_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    mask_path = os.path.join(MASK_FOLDER, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Cannot read mask: {mask_path}")
        continue

    parsed = parse_mask_filename(mask_file)
    if not parsed:
        print(f"Filename format not matched for {mask_file}, skip.")
        continue
    day, prefix = parsed

    day_folder = os.path.join(ALL_IMAGE_FOLDER, day)
    if not os.path.isdir(day_folder):
        print(f"Day folder not found: {day_folder}, skip.")
        continue

    out_day_folder = os.path.join(OUTPUT_FOLDER, day)
    os.makedirs(out_day_folder, exist_ok=True)

    # 對應 300 個波段: D0611_S1_B1_W*.png ~ B300
    for i in range(1, 301):
        band_pattern = os.path.join(day_folder, f"{prefix}_B{i}_W*.png")
        band_files = glob.glob(band_pattern)
        if not band_files:
            continue

        for bandf in band_files:
            band_img = cv2.imread(bandf, cv2.IMREAD_GRAYSCALE)
            if band_img is None:
                print(f"Cannot read band image: {bandf}")
                continue

            if band_img.shape != mask.shape:
                print(f"Size mismatch: {bandf} vs {mask_path}, skip.")
                continue

            result = cv2.bitwise_and(band_img, mask)

            # 輸出檔名: e.g. D0611_S1_B1_W388.41_masked.png
            base_name = os.path.basename(bandf)
            bbase, bext = os.path.splitext(base_name)
            out_name = f"{bbase}_masked{bext}"
            out_path = os.path.join(out_day_folder, out_name)

            cv2.imwrite(out_path, result)
            print(f"Saved: {out_path}")

print("finish")
