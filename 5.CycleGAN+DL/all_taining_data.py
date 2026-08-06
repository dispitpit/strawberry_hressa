'''
generates fake cube
output: .npy fake + RGB preview.png
'''

import os
import sys
from pathlib import Path
import numpy as np
import torch
from skimage import exposure
import matplotlib.pyplot as plt

from options.test_options import TestOptions
from models import create_model

EXPERIMENT_NAME = "strawberry_hsi_cyclegan_32bands_v2" # model
EPOCH_TAG = "latest" # latest model
DATAROOT = r"D:\Amanda_strawberry\pytorch-CycleGAN-and-pix2pix-master\datasets\strawberry_HSI"

# select 32 bands
SELECTED_BANDS = sorted({
    30,31,32,33,34,35,
    60,61,87,96,103,110,121,131,134,
    152,154,155,159,169,173,174,180,
    190,191,194,207,222,223,225,231,
    273
})
NUM_CHANNELS = len(SELECTED_BANDS)

GLOBAL_MIN, GLOBAL_MAX = np.load(
    r"D:\Amanda_strawberry\pytorch-CycleGAN-and-pix2pix-master\GLOBAL_MIN_MAX.npy"
)

PATCH_SIZE = 128
STRIDE = 64
GAMMA = 1.0

OUTPUT_ROOT = Path("./low_output_v1")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ======================================================
# function

def to_chw(arr):
    # goal : (C, H, W)
    if arr.ndim != 3:
        raise ValueError("Must be (H,W,C) or (C,H,W)")

    C = arr.shape[0]
    if C == 300 or C == len(SELECTED_BANDS):
        return arr  # Already CHW
    elif arr.shape[-1] == 300 or arr.shape[-1] == len(SELECTED_BANDS):
        return arr.transpose(2, 0, 1)
    else:
        raise RuntimeError(f"Cannot detect channel axis: {arr.shape}")

def make_tile_indices(H, W, ps, stride):
    # count the number of patch in samples
    ys, xs = [], []

    y = 0
    while y + ps <= H:
        ys.append(y)
        y += stride
    if ys[-1] != H - ps:
        ys.append(H - ps)

    x = 0
    while x + ps <= W:
        xs.append(x)
        x += stride
    if xs[-1] != W - ps:
        xs.append(W - ps)

    return ys, xs

def cube_to_rgb_fixed(cube, r, g, b, vmin, vmax, gamma):
    rgb = np.stack([cube[r], cube[g], cube[b]], axis=-1)
    rgb_norm = (rgb - vmin) / (vmax - vmin + 1e-8)
    rgb_norm = np.clip(rgb_norm, 0, 1)
    return exposure.adjust_gamma(rgb_norm, gamma=gamma)

def band_pos(band_300):
    return SELECTED_BANDS.index(band_300)

BAND_B = band_pos(30)
BAND_G = band_pos(60)
BAND_R = band_pos(121)

def load_generator(direction):
    print("start")
    """
    direction = 'AtoB' or 'BtoA'
    returns: netG, device
    """

    sys.argv = [
        "all_training_data.py",
        "--dataroot", "./dummy",
        "--name", EXPERIMENT_NAME,
        "--model", "cycle_gan",
        "--dataset_mode", "unaligned",
        "--input_nc", str(NUM_CHANNELS),
        "--output_nc", str(NUM_CHANNELS),
        "--phase", "test",
    ]

    opt = TestOptions().parse()
    opt.isTrain = False
    opt.input_nc = NUM_CHANNELS
    opt.output_nc = NUM_CHANNELS
    opt.checkpoints_dir = r"D:\Amanda_strawberry\pytorch-CycleGAN-and-pix2pix-master\checkpoints"
    opt.epoch = EPOCH_TAG
    opt.direction = direction
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    device = opt.device

    if direction == "AtoB":
        netG = model.netG_A
    else:
        netG = model.netG_B

    netG.to(device)

    print(f"Model loaded. Generator for {direction} on {device}.")
    return netG, device


# inference for one HSI file
def process_one_file(netG, device, filepath: Path, out_dir: Path):
    print(f"\nProcessing: {filepath}")

    cube = np.load(filepath).astype(np.float32)
    cube_chw_full = to_chw(cube)
    cube_sel = cube_chw_full[SELECTED_BANDS]

    C, H, W = cube_sel.shape

    # normalization
    cube_norm = (cube_sel - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + 1e-8)
    cube_norm = cube_norm * 2 - 1

    fake_cube = np.zeros((C, H, W), np.float32)
    weight = np.zeros((H, W), np.float32)

    ys, xs = make_tile_indices(H, W, PATCH_SIZE, STRIDE)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = cube_norm[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                tensor = (
                    torch.from_numpy(patch)
                    .unsqueeze(0)  # (1,C,H,W)
                    .float()
                    .to(device)
                )
                fake_patch = netG(tensor).cpu().numpy()[0]

                fake_cube[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += fake_patch
                weight[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    fake_cube /= weight[None, :, :]

    # denormalize
    fake_phys = (fake_cube + 1) / 2
    fake_phys = fake_phys * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN

    # save npy
    out_npy = out_dir / f"{filepath.stem}_fake.npy"
    np.save(out_npy, fake_phys)

    # save RGB
    rgb = cube_to_rgb_fixed(fake_phys, BAND_R, BAND_G, BAND_B,
                            vmin=GLOBAL_MIN, vmax=GLOBAL_MAX, gamma=GAMMA)
    plt.imsave(out_dir / f"{filepath.stem}_fake.png", rgb)

    print(f"Saved: {out_npy}")


# _______________________________________________________________________
def main():
    netG_AtoB, device_A = load_generator("AtoB")
    netG_BtoA, device_B = load_generator("BtoA")

    dirs = {
        "trainA": (netG_AtoB, device_A, "fakeB"),
        "trainB": (netG_BtoA, device_B, "fakeA"),
    }

    for folder_name, (netG, device, suffix) in dirs.items():
        folder = Path(DATAROOT) / folder_name
        out_dir = OUTPUT_ROOT / f"{folder_name}_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Processing {folder_name} ({suffix}) ===")

        files = sorted(folder.glob("*.npy"))
        print(f"Found {len(files)} files")

        for f in files:
            process_one_file(netG, device, f, out_dir)

    print("\nAll done.")

if __name__ == "__main__":
    main()
# _______________________________________________________________________
