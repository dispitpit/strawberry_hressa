# 新的vis建表
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


wavelengths = np.load("wavelengths.npy")

# === 所需波段對應 index（±2nm 容忍）===
def find_band_idx(w):
    diff = np.abs(wavelengths - w)
    return diff.argmin() if np.min(diff) <= 2 else None

# required_waves = [
#     800, 680, 554, 667, 445, 700, 670, 550, 678, 500, 750, 754, 709, 681,
#     860, 660, 470, 705, 740, 734, 747, 720, 531, 570, 850, 510, 900, 970,
#     1240
# ]
required_waves = [
    445, 490, 500, 510, 531, 540, 550, 554, 560, 570,
    600, 630, 660, 667, 670, 680, 690, 700, 705, 710,
    715, 720, 726, 734, 740, 747, 750, 760, 790, 800,
    810, 850, 900, 970
]
idx_map = {w: find_band_idx(w) for w in required_waves if find_band_idx(w) is not None}

print("\n=== 實際使用波段與誤差（±2nm 容忍）===")
for target_w in required_waves:
    idx = idx_map.get(target_w)
    if idx is not None:
        actual_w = wavelengths[idx]
        delta = round(actual_w - target_w, 3)
        print(f"目標: {target_w} nm  →  實際: {actual_w:.3f} nm  (誤差: {delta:+.3f} nm)")
    else:
        print(f"目標: {target_w} nm  →  無對應波段 (超出容忍範圍)")

# 建立誤差紀錄表
error_records = []
for target_w in required_waves:
    idx = idx_map.get(target_w)
    if idx is not None:
        actual_w = wavelengths[idx]
        delta = round(actual_w - target_w, 6)
        error_records.append({
            "Target Wavelength (nm)": target_w,
            "Actual Wavelength (nm)": round(actual_w, 6),
            "Delta (nm)": delta
        })
    else:
        error_records.append({
            "Target Wavelength (nm)": target_w,
            "Actual Wavelength (nm)": "Missing",
            "Delta (nm)": "N/A"
        })

# 儲存 CSV
csv_path = Path("band_matching_error_v2.csv")
error_df = pd.DataFrame(error_records)
error_df.to_csv(csv_path, index=False)


# === 建立 VI 函式庫 ===
def build_vi_functions(idx):
    def safe(f, name):
        def wrapper(R):
            try:
                return f(R)
            except Exception as e:
                print(f"[跳過 VI] {name}: {e}")
                return np.nan

        return wrapper

    return {
        "NDVI": safe(lambda R: (R[idx[800]] - R[idx[670]]) / (R[idx[800]] + R[idx[670]] + 1e-6), "NDVI"),
        "RDVI": safe(lambda R: (R[idx[800]] - R[idx[680]]) / (np.sqrt(R[idx[800]] + R[idx[680]]) + 1e-6), "RDVI"),
        "RNDVI": safe(lambda R: (R[idx[750]] - R[idx[705]]) / (R[idx[750]] + R[idx[705]] + 1e-6), "RNDVI"),
        "GI": safe(lambda R: R[idx[554]] / (R[idx[667]] + 1e-6), "GI"),
        "SIPI": safe(lambda R: (R[idx[800]] - R[idx[445]]) / (R[idx[800]] + R[idx[680]] + 1e-6), "SIPI"),
        "MCARI": safe(lambda R: ((R[idx[700]] - R[idx[670]]) - 0.2 * (R[idx[700]] - R[idx[550]])) * (R[idx[700]] / (R[idx[670]] + 1e-6)), "MCARI"),
        "MCARI2": safe(lambda R: ((R[idx[750]] - R[idx[705]]) - 0.2 * (R[idx[750]] - R[idx[550]])) * (R[idx[750]] / (R[idx[705]] + 1e-6)), "MCARI2"),
        # "MCARI2": safe(lambda R: ((R[idx[700]] - R[idx[670]]) - 0.2 * (R[idx[700]] - R[idx[550]])) * (
        #             R[idx[700]] / (R[idx[670]] + 1e-6)) * (R[idx[800]] / (R[idx[670]] + 1e-6)), "MCARI2"),
        # "PSRI": safe(lambda R: (R[idx[678]] - R[idx[500]]) / (R[idx[750]] + 1e-6), "PSRI"),
        "PSRI": safe(lambda R: (R[idx[660]] - R[idx[510]]) / (R[idx[760]] + 1e-6), "PSRI"),
        "ARI1": safe(lambda R: (1 / R[idx[550]] - 1 / R[idx[700]]), "ARI1"),
        "ARI2": safe(lambda R: R[idx[800]] * (1 / R[idx[500]] - 1 / R[idx[700]]), "ARI2"),
        "MTCI": safe(lambda R: (R[idx[750]] - R[idx[710]]) / (R[idx[710]] - R[idx[680]] + 1e-6), "MTCI"),
        "EVI": safe(lambda R: 2.5 * (R[idx[800]] - R[idx[670]]) / (R[idx[800]] + 6 * R[idx[670]] - 7.5 * R[idx[490]] + 1), "EVI"),
        "OSAVI": safe(lambda R: (1 + 0.16) * (R[idx[800]] - R[idx[670]]) / (R[idx[800]] + R[idx[670]] + 0.16), "OSAVI"),
        "MSR": safe(lambda R: (R[idx[800]] / (R[idx[670]] - 1) + 1e-6) / (np.sqrt(R[idx[800]] / (R[idx[670]] + 1e-6)) + 1e-6), "MSR"),
        # "TVI": safe(lambda R: np.sqrt((R[idx[750]] - R[idx[705]]) / (R[idx[750]] + R[idx[705]] + 1e-6)), "TVI"),
        "TVI": safe(lambda R: 0.5 * (120 * (R[idx[750]] - R[idx[550]]) - 200 * (R[idx[670]] - R[idx[550]])), "TVI"),
        "RENDVI": safe(lambda R: (R[idx[750]] - R[idx[705]]) / (R[idx[750]] + R[idx[705]] + 1e-6), "RENDVI"),
        "VOG1": safe(lambda R: R[idx[740]] / R[idx[720]], "VOG1"),
        "VOG2": safe(lambda R: (R[idx[734]] - R[idx[747]]) / (R[idx[715]] - R[idx[726]]), "VOG2"),
        "VOG3": safe(lambda R: (R[idx[734]] - R[idx[747]]) / (R[idx[715]] - R[idx[720]]), "VOG3"),
        "PRI": safe(lambda R: (R[idx[531]] - R[idx[570]]) / (R[idx[531]] + R[idx[570]] + 1e-6), "PRI"),
        "CIrededge": safe(lambda R: ((R[idx[760]] - R[idx[800]]) / (R[idx[690]] - R[idx[710]])) - 1, "CIrededge"),
        # "REP": safe(lambda R: R[idx[700]] + ((R[idx[740]] - R[idx[700]]) / 2), "REP"),
        "RVI": safe(lambda R: R[idx[810]] / (R[idx[560]] + 1e-6), "RVI"),
        "MSAVI": safe(lambda R: (2 * R[idx[800]] + 1 - np.sqrt((2 * R[idx[800]] + 1) - 8 * (R[idx[800]] - R[idx[670]]))) / 2, "MSAVI"),
        "GNDVI": safe(lambda R: (R[idx[750]] - R[idx[540]] + R[idx[570]]) / (R[idx[800]] + R[idx[540]] - R[idx[570]] + 1e-6), "GNDVI"),
        "NDRE": safe(lambda R: (R[idx[790]] - R[idx[720]]) / (R[idx[790]] + R[idx[720]] + 1e-6), "NDRE"),
        "CRI1": safe(lambda R: (1 / R[idx[510]] - 1 / R[idx[550]]), "CRI1"),
        "CRI2": safe(lambda R: (1 / R[idx[510]] - 1 / R[idx[700]]), "CRI2"),
        # "WI": safe(lambda R: R[idx[900]] / (R[idx[970]] + 1e-6), "WI"),
        "WBI": safe(lambda R: R[idx[900]] / (R[idx[970]] + 1e-6), "WBI"),
        "WSCT": safe(lambda R: (R[idx[970]] - R[idx[850]]) / (R[idx[970]] + R[idx[850]] + 1e-6), "WSCT"),
        # "NDWI": safe(lambda R: (R[idx[860]] - R[idx[1240]]) / (R[idx[860]] + R[idx[1240]] + 1e-6), "NDWI"),
        "DVI": safe(lambda R: R[idx[800]] - R[idx[680]], "DVI"),
        # "Chlorophyll Index Green": safe(lambda R: R[idx[800]] / R[idx[550]] - 1, "Chlorophyll Index Green"),
        # "Chlorophyll Index RedEdge": safe(lambda R: R[idx[800]] / R[idx[705]] - 1, "Chlorophyll Index RedEdge"),
        "TCARI": safe(lambda R: 3 * (((R[idx[700]] - R[idx[670]]) - 0.2 * (R[idx[700]] - R[idx[550]])) / (R[idx[700]] / (R[idx[670]] + 1e-6))), "TCARI"),
        # "TCARI/OSAVI": safe(lambda R: (3 * ((R[idx[700]] - R[idx[670]]) - 0.2 * (R[idx[700]] - R[idx[550]])) * (R[idx[700]] / (R[idx[670]] + 1e-6))) / ((R[idx[800]] - R[idx[680]]) / (R[idx[800]] + R[idx[680]] + 0.16)), "TCARI/OSAVI"),
        # "SAVI": safe(lambda R: ((R[idx[800]] - R[idx[680]]) * (1 + 0.5)) / (R[idx[800]] + R[idx[680]] + 0.5), "SAVI"),
        # "VARI": safe(lambda R: (R[idx[550]] - R[idx[670]]) / (R[idx[550]] + R[idx[670]] - R[idx[570]] + 1e-6), "VARI"),
        # "NDGI": safe(lambda R: (R[idx[550]] - R[idx[570]]) / (R[idx[550]] + R[idx[570]] + 1e-6), "NDGI"),
        # "RGRI": safe(lambda R: R[idx[550]] / (R[idx[670]] + 1e-6), "RGRI"),
        "FRI1": safe(lambda R: R[idx[690]] / (R[idx[630]] + 1e-6), "FRI1"),
        "FRI2": safe(lambda R: R[idx[750]] / (R[idx[800]] + 1e-6), "FRI2"),
        "FRI3": safe(lambda R: R[idx[690]] / (R[idx[600]] + 1e-6), "FRI3"),
        "FRI4": safe(lambda R: R[idx[740]] / (R[idx[800]] + 1e-6), "FRI4"),
    }


vi_funcs = build_vi_functions(idx_map)

# === 處理資料夾 ===
def process_folder(folder: Path, label: str):
    results = []
    files = list(folder.glob("*.npy"))
    for f in tqdm(files, desc=f"處理 {label} 样本"):
        try:
            spec = np.load(f)
            vi_result = {vi: func(spec) for vi, func in vi_funcs.items()}
            vi_result["File"] = f.name
            vi_result["Label"] = label
            results.append(vi_result)
        except Exception as e:
            print(f"[跳過檔案] {f.name} 錯誤: {e}")
    return results


# === 路徑設定 ===
healthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\npy\healthy")
unhealthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\npy\unhealthy")

# === 執行 ===
data = process_folder(healthy_dir, "Healthy") + process_folder(unhealthy_dir, "Unhealthy")
df = pd.DataFrame(data)
df.to_csv("vi_40_output_v2.csv", index=False)
# print("finish: output vi_40_output_v2.csv")

print("finished")