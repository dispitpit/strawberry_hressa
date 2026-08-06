from pathlib import Path
import numpy as np
import torch

# =========================================================
# config_3d.py
# 3D HSI classification configuration
#

# - 2D: 輸入 shape = (B, C, H, W)，其中 C=selected bands
# - 3D: 輸入 shape = (B, 1, D, H, W)，其中 D=selected bands
# =========================================================
# 資料設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"

# -------------------------
BASE_KS = Path(r"D:\Amanda_strawberry\ks")
# 3D 版使用同一批 HSI cube 檔案
BASE_DATA_ROOT = Path(r"D:\Amanda_strawberry\2D_crop")
AUG_BASE_ROOT = Path(r"D:\Amanda_strawberry\CycleGAN\final_result")

CONCENTRATIONS = ["all"]
# CONCENTRATIONS = ["low", "high"]

AUG_DIR_MAP = {
    "all": "final_use_all",
}

RESULTS_ROOT = Path(r"D:\Amanda_strawberry\results")
CACHE_ROOT = Path(r"D:\Amanda_strawberry\cache_3d_hsi")
PREBUILD_CACHE = True

# -------------------------
# A / B domain 設定
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

SPECTRAL_DEPTH = len(SELECTED_BANDS)
INPUT_CHANNELS_3D = 1
NUM_CLASSES = 2

# -------------------------
# 訓練參數
TARGET_HW = (128, 128)
BATCH_SIZE = 4
BATCH_SIZES = [4, 8, 16]
LR = 1e-3
EPOCHS = 50
NUM_WORKERS = 0
N_FOLDS = 5
SEED = 37
PRETRAINED = False
USE_AMP = True
CUDNN_BENCHMARK = True
RUN_COMPARE_AUG = [False, True]
MAX_FAKE_RATIO = 1.0
FINAL_EPOCH_STRATEGY = "mean_best_epoch"
MODEL_NAME = "resnet18_3d"

# -------------------------
# normalization
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
