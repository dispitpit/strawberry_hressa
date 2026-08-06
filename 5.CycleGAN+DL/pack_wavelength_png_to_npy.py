# 目的: 將高濃度和低濃度資料集的png照片轉乘npy集合，已用來訓練生成式ai
# 執行方法: 直接執行

#All done. success=340, skipped/failed=0
# Output -> D:\Users\Amanda\PycharmProjects\test\test_0911_GAI\npy
import os, re, glob
from pathlib import Path
import numpy as np
import imageio.v2 as iio
from collections import defaultdict

# ======== 參數設定 ========
BASE_ROOT = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0801_DL") # png資料來源
TARGET_DIRS = [BASE_ROOT / "high_concentration", BASE_ROOT / "low_concentration"] # 取高低濃度兩個資料夾
DST = Path(r"D:/Users/Amanda/PycharmProjects/test/test_0911_GAI") # 當前目錄
DST_ROOT = DST / "npy"             # 儲存地
BANDS_PER_SAMPLE = 300             # 每個樣本累積的波段數
NORMALIZE_PER_CUBE = False         # 轉檔時做 0~1 normalize（GAN 端會再映到 [-1,1]）
# ==========================

# 取檔名中的 W 值作排序鍵；
# 抓不到 W 改用 B；
# 再不行用檔名字典序
RE_W = re.compile(r"_W(\d+(?:\.\d+)?)")
RE_B = re.compile(r"_B(\d+)")

def wavelength_key(p: Path):
    s = p.stem # 去除檔案附檔名

    mw = RE_W.search(s)
    if mw:
        return float(mw.group(1))

    mb = RE_B.search(s)
    if mb:
        return int(mb.group(1))
    return s

# 以 _B 或 _W 之前為前綴（例：D0618_S16_B6_... -> D0618_S16）
def extract_prefix(p: Path):
    s = p.stem
    m = re.match(r"^(.*?)(?:_B\d+|_W\d+(?:\.\d+)?).*", s)
    return m.group(1) if m else s

# ===============================
#       將多張圖堆疊成HIS
# ==============================
def stack_pngs(png_list):
    arrs = [iio.imread(p).astype("float32") for p in png_list]
    x = np.stack(arrs, axis=0)
    if NORMALIZE_PER_CUBE:
        mn, mx = float(x.min()), float(x.max())
        if mx > mn:
            x = (x - mn) / (mx - mn)
    return x.astype("float32")

def process_leaf_dir(leaf_dir: Path):
    pngs = [Path(p) for p in glob.glob(str(leaf_dir / "*.png"))]
    if not pngs:
        return 0, 0, f"skip {leaf_dir} (no png)"

    groups = defaultdict(list)
    for p in pngs:
        groups[extract_prefix(p)].append(p)

    ok = fail = 0
    # ================== 輸出 ===================
    rel = leaf_dir.relative_to(BASE_ROOT)
    out_dir = DST_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    # ===========================================

    for prefix, plist in groups.items():
        plist.sort(key=wavelength_key)
        n = len(plist)
        if n < BANDS_PER_SAMPLE:
            print(f"[WARN] {leaf_dir} | {prefix}: only {n} < {BANDS_PER_SAMPLE}, skip")
            fail += 1
            continue
        if n > BANDS_PER_SAMPLE:
            print(f"[INFO]  {leaf_dir} | {prefix}: {n} > {BANDS_PER_SAMPLE}, use first {BANDS_PER_SAMPLE}")
            plist = plist[:BANDS_PER_SAMPLE]

        cube = stack_pngs(plist)
        out_path = out_dir / f"{prefix}.npy"
        np.save(out_path, cube)
        ok += 1
        print(f"[OK]   {out_path}  shape={cube.shape}")
    return ok, fail, f"done {leaf_dir}"

def main():
    total_ok = total_fail = 0
    for root in TARGET_DIRS:
        if not root.exists():
            print(f"[SKIP] {root} not found")
            continue

        for folder, subdirs, files in os.walk(root):
            d = Path(folder)
            if any(d.glob("*.png")):
                ok, fail, _ = process_leaf_dir(d)
                total_ok += ok; total_fail += fail
    print(f"\nAll done. success={total_ok}, skipped/failed={total_fail}")
    print(f"Output -> {DST_ROOT}")

if __name__ == "__main__":
    main()

# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0619_S15.npy  shape=(300, 1492, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0619_S6.npy  shape=(300, 1492, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0619_S7.npy  shape=(300, 1492, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0619_S8.npy  shape=(300, 1492, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0619_S9.npy  shape=(300, 1492, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S10.npy  shape=(300, 1494, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S11.npy  shape=(300, 1599, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S12.npy  shape=(300, 1600, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S13.npy  shape=(300, 1600, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S14.npy  shape=(300, 1600, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S15.npy  shape=(300, 1600, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S6.npy  shape=(300, 1494, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S7.npy  shape=(300, 1493, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S8.npy  shape=(300, 1494, 1920)
# [OK]   D:\Users\Amanda\PycharmProjects\test\test_0801_DL\npy\high_concentration\diseased\D0620_S9.npy  shape=(300, 1494, 1920)