from pathlib import Path
import numpy as np
import torch

# =========================================================
# config_2d.py
# 1. 一次跑 low / high
# 2. 一次 sweep batch size = [8, 16, 32, 64]
# 3. 每個組合都跑 no_aug / aug
# 4. 每個組合都輸出 ACC(Test) / ACC(CV5) / SP / SN / MCC
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"

# -------------------------
# 資料設定
BASE_KS = Path(r"D:\Amanda_strawberry\ks")
BASE_DATA_ROOT = Path(r"D:\Amanda_strawberry\2D_crop")

# 你的 CycleGAN fake 結果根目錄
AUG_BASE_ROOT = Path(r"D:\Amanda_strawberry\CycleGAN\final_result")

CONCENTRATIONS = ["all"]
# CONCENTRATIONS = ["low", "high"]
# CONCENTRATIONS = ["low"]
# CONCENTRATIONS = ["high"]

AUG_DIR_MAP = {
    "all": "final_use_all",
}

RESULTS_ROOT = Path(r"D:\Amanda_strawberry\results")
CACHE_ROOT = Path(r"D:\Amanda_strawberry\cache_2d_hsi")
PREBUILD_CACHE = True

# -------------------------
# A / B domain 設定
# A=healthy, B=unhealthy
A_LABEL = 0
B_LABEL = 1
TRAIN_AUG_DIR_A2B = "trainA_fakeB"
TRAIN_AUG_DIR_B2A = "trainB_fakeA"

# -------------------------
# band selection
WAVELENGTHS_PATH = Path(__file__).with_name("wavelengths.npy")
BAND_CENTERS = np.load(WAVELENGTHS_PATH)

SELECTED_BANDS = [
    30, 31, 32, 33, 34, 35,
    60, 61, 87, 96, 103, 110, 121, 131, 134,
    152, 154, 155, 159, 169, 173, 174, 180,
    190, 191, 194, 207, 222, 223, 225, 231,
    273
]
SELECTED_WAVELENGTHS = BAND_CENTERS[np.array(SELECTED_BANDS)]

IN_CHANNELS = len(SELECTED_BANDS)
NUM_CLASSES = 2

# -------------------------
# 訓練參數
TARGET_HW = (256, 256)
BATCH_SIZE = 16
BATCH_SIZES = [8, 16, 32, 64]
LR = 1e-3
EPOCHS = 100
NUM_WORKERS = 0
N_FOLDS = 5
SEED = 37
PRETRAINED = True
USE_AMP = True
CUDNN_BENCHMARK = True
# False -> no_aug
# True  -> aug
RUN_COMPARE_AUG = [False, True]
MAX_FAKE_RATIO = 1.0
FINAL_EPOCH_STRATEGY = "mean_best_epoch"

# -------------------------
# normalization
# 1. [scalar_min, scalar_max]
# 2. [min_array_300, max_array_300]
# 3. [min_array_32,  max_array_32]
# -------------------------
GLOBAL_MINMAX_PATH = Path(r"D:\Amanda_strawberry\CycleGAN\GLOBAL_MIN_MAX.npy")
_loaded = np.load(GLOBAL_MINMAX_PATH, allow_pickle=True)
GLOBAL_MIN, GLOBAL_MAX = _loaded

GLOBAL_MIN = np.asarray(GLOBAL_MIN, dtype=np.float32)
GLOBAL_MAX = np.asarray(GLOBAL_MAX, dtype=np.float32)
num_sel = len(SELECTED_BANDS)

if GLOBAL_MIN.ndim == 0 and GLOBAL_MAX.ndim == 0:
    NORM_MIN = np.full((num_sel,), float(GLOBAL_MIN), dtype=np.float32)
    NORM_MAX = np.full((num_sel,), float(GLOBAL_MAX), dtype=np.float32)
elif GLOBAL_MIN.ndim == 1 and len(GLOBAL_MIN) == 300:
    NORM_MIN = GLOBAL_MIN[SELECTED_BANDS]
    NORM_MAX = GLOBAL_MAX[SELECTED_BANDS]
elif GLOBAL_MIN.ndim == 1 and len(GLOBAL_MIN) == num_sel:
    NORM_MIN = GLOBAL_MIN
    NORM_MAX = GLOBAL_MAX
else:
    raise ValueError(
        "GLOBAL_MIN / GLOBAL_MAX shape 不符合預期。"
        f" GLOBAL_MIN shape={GLOBAL_MIN.shape}, GLOBAL_MAX shape={GLOBAL_MAX.shape}"
    )
