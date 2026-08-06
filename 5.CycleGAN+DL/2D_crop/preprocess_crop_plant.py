import os
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi

BASE = Path(r"D:\Amanda_strawberry\2D\all_concentration")

IN_HEALTHY   = BASE / "healthy"
IN_UNHEALTHY = BASE / "unhealthy"

OUT_BASE = Path(r"D:\Amanda_strawberry\2D_crop\all")
OUT_HEALTHY   = OUT_BASE / "healthy"
OUT_UNHEALTHY = OUT_BASE / "unhealthy"

OUT_HEALTHY.mkdir(parents=True, exist_ok=True)
OUT_UNHEALTHY.mkdir(parents=True, exist_ok=True)


def crop_plant_region(cube: np.ndarray, pad=20):
    C, H, W = cube.shape
    proj = cube.max(axis=0)

    # ---- normalize 到 [0,1] ----
    denom = np.ptp(proj)
    if denom < 1e-6:
        return cube

    proj_norm = (proj - proj.min()) / (denom + 1e-6)
    proj_u8 = (proj_norm * 255).astype("uint8")

    thresh = np.mean(proj_u8) * 0.5
    mask = proj_u8 > thresh

    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))

    labeled, nlab = ndi.label(mask)
    if nlab == 0:
        return cube

    sizes = ndi.sum(mask, labeled, index=range(1, nlab + 1))
    max_label = np.argmax(sizes) + 1
    plant_mask = (labeled == max_label)

    ys, xs = np.where(plant_mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, H - 1)
    x1 = min(x1 + pad, W - 1)

    cube_crop = cube[:, y0:y1 + 1, x0:x1 + 1]
    return cube_crop



def process_folder(in_dir: Path, out_dir: Path):
    npy_files = sorted(list(in_dir.glob("*.npy")))
    print(f"\n[INFO] Found {len(npy_files)} npy files in {in_dir}")

    for npy_path in npy_files:
        print("=" * 65)
        print(f"[INFO] Processing: {npy_path.name}")

        cube = np.load(npy_path)
        print(f"[INFO] Original cube shape: {cube.shape}")

        cube_crop = crop_plant_region(cube)
        print(f"[INFO] Cropped  cube shape: {cube_crop.shape}")

        out_path = out_dir / npy_path.name
        np.save(out_path, cube_crop)

        print(f"[INFO] --> Saved to: {out_path}")


def main():
    print("==========================================")
    print("     Stage 1 : Crop plant region          ")
    print("==========================================")

    process_folder(IN_HEALTHY, OUT_HEALTHY)
    process_folder(IN_UNHEALTHY, OUT_UNHEALTHY)

    print("\n[FINISHED] All cropping completed!")
    print(f"[OUTPUT] Saved to: {OUT_BASE}")


if __name__ == "__main__":
    main()
