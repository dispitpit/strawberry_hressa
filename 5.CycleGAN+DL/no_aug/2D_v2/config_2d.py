# config_2d.py
# 作用：
#   - 集中管理 2D HSI 分類實驗的「路徑、波段選擇、訓練超參數」。
#   - 被 train_2d.py、dataset_hsi_2d_patch.py 等模組 import。

from pathlib import Path
import numpy as np
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONCENTRATION = "high"
BASE_KS = Path(r"D:\Amanda_strawberry\ks")

# =============ks第二版本結果===============
TRAIN_CSV = (
    BASE_KS
    / f"ks_{CONCENTRATION}_divided_result"
    / f"ks_split_train_{CONCENTRATION}.csv"
)

TEST_CSV = (
    BASE_KS
    / f"ks_{CONCENTRATION}_divided_result"
    / f"ks_split_test_{CONCENTRATION}.csv"
)
# =========================================

# --- .npy 影像 ---
# 你的實際路徑：
#   D:\Users\Amanda\PycharmProjects\test\test_0911_GAI\cropped\high\healthy
#   D:\Users\Amanda\PycharmProjects\test\test_0911_GAI\cropped\high\unhealthy (之後會有)
BASE_DATA_ROOT = Path(r"D:\Amanda_strawberry\2D_crop")

BASE_DATA = BASE_DATA_ROOT / CONCENTRATION  # => cropped/high

WAVELENGTHS_PATH = Path(__file__).with_name("wavelengths.npy")
BAND_CENTERS = np.load(WAVELENGTHS_PATH)   # shape = (300,)

SELECTED_BANDS = [
    30, 31, 32, 33, 34, 35,
    60, 61, 87, 96, 103, 110, 121, 131, 134,
    152, 154, 155, 159, 169, 173, 174, 180,
    190, 191, 194, 207, 222, 223, 225, 231,
    273
]

SELECTED_WAVELENGTHS = BAND_CENTERS[np.array(SELECTED_BANDS)]
# print("[INFO] Selected band indices:", SELECTED_BANDS)
# print("[INFO] Selected wavelengths (nm):", np.round(SELECTED_WAVELENGTHS, 2))

# SELECTED_BANDS = [
#     30, 31, 32, 33, 34, 35,
#     60, 61, 87, 96, 103, 110, 121, 131, 134,
#     152, 154, 155, 159, 169, 173, 174, 180,
#     190, 191, 194, 207, 222, 223, 225, 231,
#     273
# ]
# SELECTED_WAVELENGTHS = np.array([
#     # LDA
#     669.53, 606.21, 663.43, 620.57, 591.80,
#     # VIs
#     740.27, 720.12, 790.50, 750.33, 710.03, 705.99,
#     # PCA
#     848.63, 846.63, 782.48, 784.49, 864.67, 852.64,
#     # SPA
#     748.32, 516.91,
#     # RF
#     455.45, 464.00, 461.87, 459.73, 457.59, 453.31,
#     # CARS + LASSO
#     453.31, 519.01, 573.21, 643.06, 712.05, 762.40, 816.57, 949.03
# ], dtype=float)

# TOL = 0.5  # ±0.5 nm 容忍

'''def find_band_indices(selected_wavelengths: np.ndarray,
                      band_centers: np.ndarray,
                      tol: float):
    """
    將「目標波長列表」映射到實際 band index（0~len(band_centers)-1）。
    對每個目標波長 lam：
      - 找出 band_centers 中最接近 lam 的 index
      - 若距離 <= tol，則納入 index
      - 若距離 > tol，列出警告
    """
    indices = []
    for lam in selected_wavelengths:
        diffs = np.abs(band_centers - lam)
        idx = int(diffs.argmin())
        if diffs[idx] <= tol:
            indices.append(idx)
        else:
            print(
                f"[WARN] λ={lam:.2f} nm 找不到 ±{tol} nm 內的 band，"
                f"最近的是 index={idx} (λ={band_centers[idx]:.2f} nm, Δ={diffs[idx]:.2f})，將略過"
            )
    # 去重、排序，避免重複 band
    indices = sorted(set(indices))
    # print(f"[INFO] 共對應到 {len(indices)} 個 band index：{indices}")
    return indices
'''


# SELECTED_BANDS = find_band_indices(SELECTED_WAVELENGTHS, BAND_CENTERS, TOL)
# SELECTED_BANDS = None


# 輸入通道數
IN_CHANNELS = len(SELECTED_BANDS) if SELECTED_BANDS is not None else 300

# 分類類別數
NUM_CLASSES = 2

# 訓練相關參數
BATCH_SIZE  = 64
LR          = 1e-3
EPOCHS      = 50
# DEVICE      = "cuda"
NUM_WORKERS = 4

# 交叉驗證
N_FOLDS = 5
SEED    = 42