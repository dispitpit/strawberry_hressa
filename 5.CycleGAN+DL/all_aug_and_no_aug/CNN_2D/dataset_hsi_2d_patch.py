from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def to_chw(arr):
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    if arr.shape[0] in (32, 300):
        return arr
    if arr.shape[-1] in (32, 300):
        return arr.transpose(2, 0, 1)
    raise RuntimeError(f"Cannot detect channel axis: {arr.shape}")


class HSI2DDataset(Dataset):
    """
    whole-image 2D HSI classification dataset

    支援：
    1. real 300-band -> selected bands
    2. fake 32-band  -> 直接用
    3. normalization
    4. resize 到固定 TARGET_HW
    5. disk cache：第一次前處理後存 .pt，後續直接讀
    """

    def __init__(
        self,
        records_df,
        selected_bands,
        norm_min=None,
        norm_max=None,
        target_hw=None,
        cache_dir=None,
        use_cache=False,
    ):
        self.df = records_df.reset_index(drop=True).copy()
        self.selected_bands = list(selected_bands)
        self.num_selected = len(self.selected_bands)
        self.target_hw = tuple(target_hw) if target_hw is not None else None
        self.use_cache = bool(use_cache)

        self.norm_min = None if norm_min is None else np.asarray(norm_min, dtype=np.float32)
        self.norm_max = None if norm_max is None else np.asarray(norm_max, dtype=np.float32)

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _make_cache_key(self, src_path: str) -> str:
        hw_tag = "none" if self.target_hw is None else f"{self.target_hw[0]}x{self.target_hw[1]}"
        raw = f"{src_path}|hw={hw_tag}|bands={','.join(map(str, self.selected_bands))}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, src_path: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is None but cache path was requested")
        key = self._make_cache_key(src_path)
        stem = Path(src_path).stem
        return self.cache_dir / f"{stem}_{key}.pt"

    def _load_cube_raw(self, path: str):
        cube = np.load(path).astype(np.float32)
        cube = to_chw(cube)

        if cube.shape[0] == 300:
            cube = cube[self.selected_bands]
        elif cube.shape[0] == self.num_selected:
            pass
        else:
            raise ValueError(f"Unexpected channel count {cube.shape[0]} for {path}")

        if self.norm_min is not None and self.norm_max is not None:
            cube = (cube - self.norm_min[:, None, None]) / (
                (self.norm_max - self.norm_min)[:, None, None] + 1e-8
            )

        x = torch.from_numpy(cube).float()

        if self.target_hw is not None:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return x.contiguous()

    def get_or_build_cache(self, src_path: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is None but get_or_build_cache() was called")

        cpath = self._cache_path(src_path)
        if cpath.exists():
            return cpath

        x = self._load_cube_raw(src_path)
        torch.save(x, cpath)
        return cpath

    def _load_x(self, src_path: str):
        if self.use_cache and self.cache_dir is not None:
            cpath = self.get_or_build_cache(src_path)
            x = torch.load(cpath, map_location="cpu")
            return x.float()
        return self._load_cube_raw(src_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_path = str(row["path"])
        x = self._load_x(src_path)
        y = int(row["label"])
        return x, y
