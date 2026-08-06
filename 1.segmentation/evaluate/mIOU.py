import os
import cv2
import numpy as np

# ========== 你可以在此修改路徑 ==========

PRED_FOLDER = r"C:\Users\Amanda\PycharmProjects\test\RGB image\count_mIOU\cleaned_result"  # 模型預測結果
GT_FOLDER   = r"C:\Users\Amanda\PycharmProjects\test\RGB image\count_mIOU\train_set"          # 真實GT

# ========== 以下函式用於計算單張影像的IoU(前景/背景) ==========

def compute_image_iou(pred_gray, gt_gray):
    """
    對單張影像計算:
      - IoU_fg (前景=1)
      - IoU_bg (背景=0)
      - mIoU = (IoU_fg + IoU_bg)/2
    pred_gray, gt_gray: shape=(H,W)，可能是0/255；程式內會二值化為0/1
    回傳 (iou_fg, iou_bg, miou)
    """
    # 1. binarize: (0,255) -> (0,1)
    pred_bin = (pred_gray > 128).astype(np.uint8)
    gt_bin   = (gt_gray   > 128).astype(np.uint8)

    # 2. 前景IoU
    intersection_fg = np.logical_and(pred_bin, gt_bin).sum()
    union_fg        = np.logical_or(pred_bin, gt_bin).sum()
    if union_fg == 0:
        iou_fg = 1.0  # 若都沒前景像素, 視需求可設為1或0
    else:
        iou_fg = intersection_fg / union_fg

    # 3. 背景IoU
    intersection_bg = np.logical_and(1 - pred_bin, 1 - gt_bin).sum()
    union_bg        = np.logical_or (1 - pred_bin, 1 - gt_bin).sum()
    if union_bg == 0:
        iou_bg = 1.0
    else:
        iou_bg = intersection_bg / union_bg

    # 4. mIoU
    miou = (iou_fg + iou_bg) / 2
    return iou_fg, iou_bg, miou

def main():
    pred_files = os.listdir(PRED_FOLDER)

    iou_fg_list = []
    iou_bg_list = []
    miou_list   = []

    count = 0

    for filename in pred_files:
        pred_path = os.path.join(PRED_FOLDER, filename)
        gt_path   = os.path.join(GT_FOLDER, filename)  # 假設同名

        if not os.path.exists(gt_path):
            print(f"[警告] 找不到對應的 GT: {filename}")
            continue

        # 讀取 (灰階)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
        if pred_img is None or gt_img is None:
            print(f"[警告] 讀取失敗: {filename}")
            continue

        # shape檢查
        if pred_img.shape != gt_img.shape:
            print(f"[警告] 尺寸不匹配: {filename}")
            continue

        # 計算 IoU(前景), IoU(背景), mIoU
        iou_fg, iou_bg, miou = compute_image_iou(pred_img, gt_img)

        iou_fg_list.append(iou_fg)
        iou_bg_list.append(iou_bg)
        miou_list.append(miou)
        count += 1

        print(f"{filename}: IoU_fg={iou_fg:.4f}, IoU_bg={iou_bg:.4f}, mIoU={miou:.4f}")

    if count == 0:
        print("沒有成功比對的檔案，無法計算 mIoU。")
        return

    # 最後對所有影像的 IoU 做平均 (image-wise 平均)
    avg_iou_fg = np.mean(iou_fg_list)
    avg_iou_bg = np.mean(iou_bg_list)
    avg_miou   = np.mean(miou_list)

    print("\n======== 統計結果 ========")
    print(f"處理張數: {count}")
    print(f"平均 IoU(前景) = {avg_iou_fg:.4f}")
    print(f"平均 IoU(背景) = {avg_iou_bg:.4f}")
    print(f"平均 mIoU      = {avg_miou:.4f}")

if __name__ == "__main__":
    main()
