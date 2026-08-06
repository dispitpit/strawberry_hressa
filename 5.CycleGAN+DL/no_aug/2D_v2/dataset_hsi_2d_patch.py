# dataset_hsi_2d_patch.py
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

PATCH_SIZE = 128

class HSIDataset2D_Patch(Dataset):
    """
    128×128 Patch-Based Dataset for HSI Classification.
    - 每個 patch 都繼承整張影像的 label。
    - 第一次計算 patch 時會順便存成 .npy，之後直接從快取讀取。
    - 建立 patch index 時使用 mmap，只讀檔案 metadata，不讀整個大 cube。
    """

    def __init__(self,
                 split_csv,
                 data_root,
                 selected_bands=None,
                 transform=None,
                 file_col="File",
                 label_col="Label"):

        self.data_root = Path(data_root)
        self.selected_bands = selected_bands
        self.transform = transform

        df = pd.read_csv(split_csv)

        self.files = df[file_col].astype(str).tolist()
        self.labels_str = df[label_col].astype(str).tolist()

        # label mapping
        mapping = {"Healthy": 0, "healthy": 0,
                   "Unhealthy": 1, "unhealthy": 1}
        self.labels = [mapping[x] for x in self.labels_str]

        # patch
        import hashlib
        sig = hashlib.md5(np.asarray(self.selected_bands, np.int32).tobytes()).hexdigest()[:8]
        self.cache_root = self.data_root / f"patch_cache_{PATCH_SIZE}_C32_{sig}"
        # self.cache_root = self.data_root / f"patch_cache_{PATCH_SIZE}"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        #  建立 patch 索引
        # ------------------------------------------------------------
        print("[INFO] 建立 patch 索引（使用 mmap，不讀整張大 cube）...")
        num_files = len(self.files)
        print(f"[INFO] CSV 中共有 {num_files} 個樣本檔案。")

        self.patch_index = []

        for i, fname in enumerate(self.files):
            label_str = self.labels_str[i]
            path = self._cube_path(fname, label_str)

            # 不載入整個 cube
            cube = np.load(path, mmap_mode="r")
            C, H, W = cube.shape

            # 更新 channel 數量資訊
            if self.selected_bands is not None:
                C = len(self.selected_bands)

            nH = math.ceil(H / PATCH_SIZE)
            nW = math.ceil(W / PATCH_SIZE)

            for ph in range(nH):
                for pw in range(nW):
                    self.patch_index.append((i, ph, pw))

            if (i + 1) == 1 or (i + 1) == num_files or (i + 1) % 5 == 0:
                print(f"[INFO] 建立索引進度：{i+1}/{num_files} 個檔案已處理完畢。")

        print(f"[INFO] 共有 {len(self.patch_index)} 個 patch（索引建立完成）。")

        # 檢查快取
        if any(self.cache_root.rglob("*.npy")):
            print(f"[INFO] 偵測到既有 patch 快取：{self.cache_root}")
            print("       本次將優先從快取讀取 patch，不再重算。")
        else:
            print(f"[INFO] 尚未找到 patch 快取，第一次會自動建立於：{self.cache_root}")

        self._printed_first_cache_write = False

    # ----------------- 工具函式 -----------------
    def _cube_path(self, fname, label_str):
        """
        由 CSV 的 File + Label 推回實際 .npy 路徑。
        CSV: Label = 'Healthy' / 'Unhealthy'
        資料夾: 'healthy' / 'unhealthy'  → 這裡用 lower() 對齊
        """
        base = fname.replace(".npy", "").replace("_mean_spectrum", "")
        folder = label_str.lower()
        return self.data_root / folder / (base + ".npy")

    def _patch_cache_path(self, fname, label_str, ph, pw):
        """
        決定 patch 快取的位置：
          cache_root / <healthy|unhealthy> / <檔名身分> / phX_pwY.npy
        """
        stem = fname.replace(".npy", "").replace("_mean_spectrum", "")
        folder = label_str.lower()
        img_cache_dir = self.cache_root / folder / stem
        img_cache_dir.mkdir(parents=True, exist_ok=True)
        return img_cache_dir / f"ph{ph}_pw{pw}.npy"

    # ----------------- Dataset 介面 -----------------
    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        image_idx, ph, pw = self.patch_index[idx]

        fname = self.files[image_idx]
        label_str = self.labels_str[image_idx]
        label_val = self.labels[image_idx]

        cache_path = self._patch_cache_path(fname, label_str, ph, pw)

        # ===== 1) cache hit =====
        if cache_path.exists():
            patch_np = np.load(cache_path).astype("float32")

        else:
            # ===== 2) cache miss =====
            cube = np.load(self._cube_path(fname, label_str)).astype("float32")  # (C,H,W)
            C = cube.shape[0]

            # --- 統一通道：300->32；32 直接用 ---
            if C == 300:
                if self.selected_bands is None:
                    raise ValueError(f"C=300 but selected_bands is None. file={fname}")
                cube = cube[self.selected_bands, :, :]  # -> (32,H,W)
            elif C == 32:
                pass
            else:
                raise ValueError(f"Unexpected bands={C}, expect 300 or 32. file={fname}")

            C, H, W = cube.shape
            cube_t = torch.from_numpy(cube)

            padH = max(0, (ph + 1) * PATCH_SIZE - H)
            padW = max(0, (pw + 1) * PATCH_SIZE - W)

            cube_t = F.pad(cube_t, (0, padW, 0, padH), mode="constant", value=0)

            h0 = ph * PATCH_SIZE
            w0 = pw * PATCH_SIZE
            patch_t = cube_t[:, h0:h0 + PATCH_SIZE, w0:w0 + PATCH_SIZE].contiguous()

            patch_np = patch_t.numpy().astype("float32")
            np.save(cache_path, patch_np)

            if not self._printed_first_cache_write:
                print(f"[INFO] 第一次寫入 patch 快取：{cache_path}")
                print("       之後相同 patch 會直接從快取讀取。")
                self._printed_first_cache_write = True

        patch = torch.from_numpy(patch_np)

        if self.transform is not None:
            arr = patch.permute(1, 2, 0).numpy()
            aug = self.transform(image=arr)
            arr = aug["image"]

            if isinstance(arr, np.ndarray):
                patch = torch.from_numpy(arr.transpose(2, 0, 1)).float()
            else:
                patch = arr.float()
        else:
            patch = patch.float()

        label = torch.tensor(label_val, dtype=torch.long)
        return patch, label, image_idx

