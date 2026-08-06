# 建立HIS資料集
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import spectral
from pathlib import Path
import os
spectral.settings.envi_support_nonlowercase_params = True

#*************#
### 選擇日期
day_folders = [
    "0709",
]
### 選擇檔案號碼
start = 13
end = 15
#*************#

# 資料路徑
datapath_root = Path(fr'C:\Users\Amanda\PycharmProjects\test\病害高光譜2024June')
# 專案路徑
datapath_project = Path(r'C:\Users\Amanda\PycharmProjects\test\all_image')

# 關閉顯示圖片
plt.ioff()

for day in day_folders:
    datapath_data = datapath_root / f"2024{day}"
    output_dir = datapath_project / f"D{day}"
    os.makedirs(output_dir, exist_ok=True)

    # sample1~n_RT檔
    for i in range(start,end+1):
        # 載入檔案
        hdr_file = datapath_data / f'sample{i}_RT.hdr'
        raw_file = datapath_data / f'sample{i}_RT.raw'
        if not hdr_file.exists():
            print(f"hdr_file not exist :{hdr_file}")
            continue
        if not raw_file.exists():
            print(f"raw_file not exist :{raw_file}")
            continue

        # 開啟ENVI檔案
        image = envi.open(str(hdr_file))

        # 檢查
        # 列印元數據(看變數名稱)
        # print(image.metadata)
        ##########################################
        # 檔案處理
        # bands的wavelengths
        wavelengths = image.metadata.get('Wavelength', None)
        if wavelengths is None:
            print(f"沒找到wavelengths欄位，跳過index: {i}")
            continue

        try:
            wavelengths = [float(w) for w in wavelengths]
        except ValueError:
            print(f"wavelengths轉換失敗，跳過index: {i}")
            continue
        if len(wavelengths) != image.shape[-1]:
            print(f"bands和wavelength不一致，跳過index: {i}")
            continue
        ##########################################

        # 執行轉image部分
        for index in range(image.shape[-1]):
            try:
                band_data = image.read_band(index)
                # 正規化
                band_min = band_data.min()
                band_max = band_data.max()
                band_normalized = (band_data - band_min)/(band_max - band_min)

                # 波波波
                wavelength_value = wavelengths[index]
                # 存PNG黨
                #S -> sample
                #B -> bands index
                #W -> wavelengths
                output_path = output_dir / f'D{day}_S{i}_B{index + 1}_W{wavelength_value:.2f}.png'
                plt.imsave(str(output_path),band_normalized, cmap='gray')
                print(f"success:{output_path}")
            except Exception as e:
                print(f"fail, error:{e}")
                print(f"跳過sample{i}")
                break

        # print(f"bands :300 儲存成功")

    print('finish!')