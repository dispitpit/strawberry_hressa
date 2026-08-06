# 1001復盤ok
# 將vi_40_output.csv的結果區分成Low和hight
# result:
# vi_40_high.csv
# vi_40_low.csv
from pathlib import Path
import pandas as pd

# 讀取原始 vi_40_output.csv
df = pd.read_csv(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\vi_40_output.csv")

# 指定 high 資料夾下的 healthy 和 unhealthy 資料夾
low_healthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\low\healthy")
low_unhealthy_dir = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\low\unhealthy")

# 收集 high 資料夾中的所有檔名
low_files = {f.name for f in low_healthy_dir.glob("*.npy")} | {f.name for f in low_unhealthy_dir.glob("*.npy")}

# 篩選出符合的檔案
df_low = df[df["File"].isin(low_files)]

# 儲存
output_path = Path(r"C:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\vi_40_low.csv")
df_low.to_csv(output_path, index=False)

print("finished")