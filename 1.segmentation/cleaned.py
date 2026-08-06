# 對於mask的鋸齒或其他地方進行後續的微調
# u2net_result_mask -> u2net_result_mask_feathered
import cv2
import numpy as np
import os

# === config ============================================
MORPH_KERNEL_SIZE = (3, 3)        # 形態學的 kernel 大小
MIN_AREA = 6000                   # 面積閾值，小於此值視為雜訊
USE_MORPH = False                 # 是否使用形態學
USE_CONNECTED_COMPONENT = True    # 是否使用連通區域分析
# ========================================================

# 輸入(原遮罩)資料夾
input_folder = r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask"
# 輸出(處理後遮罩)資料夾
output_folder = r"C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask_cleaned"
os.makedirs(output_folder, exist_ok=True)

def remove_small_components(binary_mask, min_area=50):
    """
    使用連通區域分析，移除面積小於 min_area 的前景(白色)區域
    binary_mask: 二值遮罩 (0/255)
    min_area: 面積閾值，小於此值的白區域視為雜訊去除
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for label_idx in range(1, num_labels):  # label=0 是背景
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 255
    return cleaned

valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_exts):
        input_path = os.path.join(input_folder, filename)

        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: cannot read {input_path}, skipped.")
            continue

        # ---------- 開運算去雜訊 ----------
        final_mask = mask
        if USE_MORPH:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
            # 開運算
            opened = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            final_mask = opened

        # ---------- 連通區域分析 ----------
        if USE_CONNECTED_COMPONENT:
            final_mask = remove_small_components(final_mask, min_area=MIN_AREA)

        # ---------- 儲存 ----------
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final_mask)
        # print(f"Saved cleaned mask: {output_path}")

print("finish")
