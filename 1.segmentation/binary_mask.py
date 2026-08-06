# 轉換圖片確保為mask的形式 only 0/255
# u2net_result -> u2net_result_mask
import cv2
import numpy as np
import os

# 輸入影像資料夾 (彩色葉子黑背景)
input_folder = r'C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results'
# 輸出遮罩資料夾 (單通道 0/255)
output_folder = r'C:\Users\Amanda\PycharmProjects\test\RGB image\u2net_final_results_mask'


os.makedirs(output_folder, exist_ok=True)

# 遍歷 input_folder 下的所有影像檔
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
        input_path = os.path.join(input_folder, filename)

        # 讀取彩色影像 (BGR)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Cannot read file {input_path}. Skipped.")
            continue

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        non_black = np.any(img != [0, 0, 0], axis=-1)
        mask[non_black] = 255

        # 檔名: 原檔名 + "_mask.png"
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_mask.png"
        output_path = os.path.join(output_folder, output_filename)

        cv2.imwrite(output_path, mask)
        print(f"Saved mask: {output_path}")

print("finish")
