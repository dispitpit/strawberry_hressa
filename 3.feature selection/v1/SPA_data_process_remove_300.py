import numpy as np
from pathlib import Path


wavelength_path = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\wavelengths.npy")
output_wavelength_path = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\wavelengths_remove_bands_300.npy")

healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy_SPA\healthy")
diseased_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0415_LDA\npy_SPA\unhealthy")

wavelengths = np.load(wavelength_path)
mask = wavelengths >= 400
filtered_wavelengths = wavelengths[mask]

np.save(output_wavelength_path, filtered_wavelengths)
print(f"[訊息] 新的 wavelengths 檔案已儲存為: {output_wavelength_path}")





def process_npy_files(directory, mask):
    files = list(directory.glob("*.npy"))
    for file in files:
        data = np.load(file)
        if len(data) != len(wavelengths):
            print(f"[警告] {file.name} 的資料長度與 wavelengths 不符，略過。")
            continue
        filtered_data = data[mask]
        np.save(file, filtered_data)
        print(f"[完成] 已處理: {file.name}")


# print("\n[處理 healthy 資料夾]")
process_npy_files(healthy_dir, mask)

# print("\n[處理 diseased 資料夾]")
process_npy_files(diseased_dir, mask)

print("finish")
