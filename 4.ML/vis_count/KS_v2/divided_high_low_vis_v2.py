# 新的vis!!!
# 將vi_40_output.csv的結果區分成Low和hight
# result:
# vi_40_high_v2.csv
# vi_40_low_v2.csv
from pathlib import Path
import pandas as pd

# 讀取原始 vi_40_output.csv
df = pd.read_csv(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\new_vis1105\vi_40_output_v2.csv")
# 指定 high 資料夾下的 healthy 和 unhealthy 資料夾
low_healthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\high\healthy")
low_unhealthy_dir = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\high\unhealthy")

low_files = {f.name for f in low_healthy_dir.glob("*.npy")} | {f.name for f in low_unhealthy_dir.glob("*.npy")}

df_low = df[df["File"].isin(low_files)]

output_path = Path(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\new_vis1105\KS_v2\vi_40_high_v2.csv")
df_low.to_csv(output_path, index=False)

print("finished")