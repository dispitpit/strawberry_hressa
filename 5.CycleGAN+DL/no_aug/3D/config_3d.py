# config_3d.py
from pathlib import Path
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============ Experiment ============
CONCENTRATION = "high"  # low / high
BASE_KS = Path(r"D:\Amanda_strawberry\ks")

TRAIN_CSV = BASE_KS / f"ks_{CONCENTRATION}_divided_result" / f"ks_split_train_{CONCENTRATION}.csv"
TEST_CSV  = BASE_KS / f"ks_{CONCENTRATION}_divided_result" / f"ks_split_test_{CONCENTRATION}.csv"

# data_root should contain:
#   <data_root>/<low|high>/healthy/*.npy
#   <data_root>/<low|high>/unhealthy/*.npy
BASE_DATA_ROOT = Path(r"D:\Amanda_strawberry\2D_crop")
BASE_DATA = BASE_DATA_ROOT / CONCENTRATION

# ============ Bands ============
# Select 32 bands
SELECTED_BANDS = sorted(set([
    30, 31, 32, 33, 34, 35,
    60, 61, 87, 96, 103, 110, 121, 131, 134,
    152, 154, 155, 159, 169, 173, 174, 180,
    190, 191, 194, 207, 222, 223, 225, 231,
    273,
]))
SPECTRAL_DEPTH = len(SELECTED_BANDS)

WAVELENGTHS_PATH = Path(__file__).with_name("wavelengths.npy")
if WAVELENGTHS_PATH.exists():
    BAND_CENTERS = np.load(WAVELENGTHS_PATH)
    SELECTED_WAVELENGTHS = BAND_CENTERS[np.array(SELECTED_BANDS)]
else:
    SELECTED_WAVELENGTHS = None

# ============ Patch / Train ============
PATCH_SIZE  = 128
BATCH_SIZE  = 4      # 3D 通常比 2D 吃 VRAM
LR          = 1e-3
EPOCHS      = 100
NUM_WORKERS = 4

NUM_CLASSES = 2
SEED        = 20
