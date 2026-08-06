import cv2
import numpy as np
import os

# 遮罩資料夾
mask_folder = r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask_cleaned"
# 原高光譜影像資料夾
original_folder = r"C:\Users\Amanda\PycharmProjects\test\RGB image\all"
# 輸出疊合結果資料夾
output_folder = r"C:\Users\Amanda\PycharmProjects\test\RGB image\overlay_final_results"
os.makedirs(output_folder, exist_ok=True)


valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

for filename in os.listdir(mask_folder):
    if not filename.lower().endswith(valid_exts):
        continue

    mask_path = os.path.join(mask_folder, filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: cannot read mask {mask_path}")
        continue

    # 假設檔名為 D0611_S1_mask.png
    # 去除 "_mask" 字樣得到 "D0611_S1"
    # 以尋找對應的原圖
    name, ext = os.path.splitext(filename)  # e.g. name="D0611_S1_mask", ext=".png"
    if "_mask" in name:
        original_name = name.replace("_mask", "")  # "D0611_S1"
    else:
        original_name = name

    # 在原圖資料夾尋找相同名稱 + 副檔
    possible_original = os.path.join(original_folder, original_name + ext)

    if not os.path.exists(possible_original):
        print(f"Warning: cannot find original for {filename}, tried {possible_original}")
        continue

    original = cv2.imread(possible_original, cv2.IMREAD_COLOR)
    if original is None:
        print(f"Warning: cannot read original {possible_original}")
        continue

    if original.shape[:2] != mask.shape[:2]:
        print(f"Size mismatch: {possible_original} vs {mask_path}. Skipped.")
        continue

    # ========== 紅色疊合 (半透明) ==========
    # 建立一張同樣大小的彩色圖, 將 mask=255 的地方填成 (0,0,255)
    overlay_color = np.zeros_like(original, dtype=np.uint8)
    overlay_color[mask == 255] = (0, 0, 255)  # 紅色 (B,G,R)

    # 使用 addWeighted 進行疊合: overlay_color 以 alpha=0.5 疊加在原圖上
    alpha = 0.5
    overlaid = cv2.addWeighted(original, 1.0, overlay_color, alpha, 0)

    # ========== bitwise_and (只保留前景, 背景=黑) ==========
    # result = cv2.bitwise_and(original, original, mask=mask)

    # 輸出檔案
    out_filename = f"{original_name}_overlay{ext}"
    out_path = os.path.join(output_folder, out_filename)

    cv2.imwrite(out_path, overlaid)
    # print(f"Saved overlay to: {out_path}")

print("All done!")
