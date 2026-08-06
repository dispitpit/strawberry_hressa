# dataset_hsi_3d_patch.py
import math
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class HSIDataset3D_Patch(Dataset):
    """
    3D Patch Dataset for HSI (Conv3d route A)
    - spatial patch: (H,W) = PATCH_SIZE
    - spectral depth: D = 32 (or selected bands)
    - output patch tensor: (1, D, H, W) for Conv3d
    """

    def __init__(
        self,
        split_csv,
        data_root,
        patch_size: int,
        selected_bands,
        transform=None,
        file_col="File",
        label_col="Label",
        verbose_every=5,
    ):
        self.data_root = Path(data_root)
        self.patch_size = int(patch_size)
        self.selected_bands = selected_bands
        self.transform = transform

        if self.selected_bands is None:
            raise ValueError("3D route expects selected_bands (e.g., 32-band indices).")

        df = pd.read_csv(split_csv)
        self.files = df[file_col].astype(str).tolist()
        self.labels_str = df[label_col].astype(str).tolist()

        mapping = {"Healthy": 0, "healthy": 0, "Unhealthy": 1, "unhealthy": 1}
        self.labels = [mapping[x] for x in self.labels_str]

        # cache root
        sig = hashlib.md5(np.asarray(self.selected_bands, np.int32).tobytes()).hexdigest()[:8]
        self.cache_root = self.data_root / f"patch_cache3d_ps{self.patch_size}_D{len(self.selected_bands)}_{sig}"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # build patch index
        print("[INFO] Building 3D patch index (mmap header only)...")
        num_files = len(self.files)
        print(f"[INFO] CSV samples: {num_files}")

        self.patch_index = []
        for i, fname in enumerate(self.files):
            label_str = self.labels_str[i]
            path = self._cube_path(fname, label_str)

            cube = np.load(path, mmap_mode="r")
            C, H, W = cube.shape

            nH = math.ceil(H / self.patch_size)
            nW = math.ceil(W / self.patch_size)

            # 3D
            for ph in range(nH):
                for pw in range(nW):
                    self.patch_index.append((i, ph, pw))

            if (i + 1) == 1 or (i + 1) == num_files or (i + 1) % verbose_every == 0:
                print(f"[INFO] Index progress: {i+1}/{num_files}")

        print(f"[INFO] Total patches: {len(self.patch_index)}")

        if any(self.cache_root.rglob("*.npy")):
            print(f"[INFO] Found existing 3D patch cache: {self.cache_root}")
            print("       Will read from cache first.")
        else:
            print(f"[INFO] No 3D patch cache yet. Will build at: {self.cache_root}")

        self._printed_first_cache_write = False

    def _cube_path(self, fname, label_str):
        base = fname.replace(".npy", "").replace("_mean_spectrum", "")
        folder = label_str.lower()
        return self.data_root / folder / (base + ".npy")

    def _patch_cache_path(self, fname, label_str, ph, pw):
        stem = fname.replace(".npy", "").replace("_mean_spectrum", "")
        folder = label_str.lower()
        img_cache_dir = self.cache_root / folder / stem
        img_cache_dir.mkdir(parents=True, exist_ok=True)
        return img_cache_dir / f"ph{ph}_pw{pw}.npy"

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        image_idx, ph, pw = self.patch_index[idx]

        fname = self.files[image_idx]
        label_str = self.labels_str[image_idx]
        label_val = self.labels[image_idx]

        cache_path = self._patch_cache_path(fname, label_str, ph, pw)

        if cache_path.exists():
            patch_np = np.load(cache_path).astype("float32")
        else:
            cube = np.load(self._cube_path(fname, label_str)).astype("float32")
            C0 = cube.shape[0]

            # unify channels: 300->32, 32 keep
            if C0 == 300:
                cube = cube[self.selected_bands, :, :]
            elif C0 == len(self.selected_bands):
                pass
            else:
                raise ValueError(f"Unexpected C={C0}. Expect 300 or {len(self.selected_bands)}. file={fname}")

            D, H, W = cube.shape
            cube_t = torch.from_numpy(cube)

            padH = max(0, (ph + 1) * self.patch_size - H)
            padW = max(0, (pw + 1) * self.patch_size - W)
            cube_t = F.pad(cube_t, (0, padW, 0, padH), mode="constant", value=0)

            h0 = ph * self.patch_size
            w0 = pw * self.patch_size
            patch_t = cube_t[:, h0:h0 + self.patch_size, w0:w0 + self.patch_size].contiguous()

            patch_np = patch_t.numpy().astype("float32")
            np.save(cache_path, patch_np)

            if not self._printed_first_cache_write:
                print(f"[INFO] First 3D cache write: {cache_path}")
                self._printed_first_cache_write = True

        patch = torch.from_numpy(patch_np).unsqueeze(0).float()

        label = torch.tensor(label_val, dtype=torch.long)
        return patch, label, torch.tensor(image_idx, dtype=torch.long)
