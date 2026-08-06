# 前處理: 300nm前有去除
#
# === CARS 結果 ===
# 最佳 CV R^2: -0.2105
# 選到的波段索引: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#   18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
#   36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
#   54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
#   72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
#   90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
#  108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
#  126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
#  144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
#  162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
#  180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
#  198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
#  216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
#  234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
#  252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269
#  270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287
#  288 289 290 291 292 293]
# 選到的實際波長: [ 401.504453  403.681314  405.856545  408.030155  410.202151  412.372541
#   414.541332  416.708532  418.87415   421.038191  423.200665  425.361579
#   427.52094   429.678757  431.835037  433.989787  436.143016  438.294732
#   440.444941  442.593651  444.740871  446.886608  449.030869  451.173663
#   453.314997  455.454879  457.593316  459.730317  461.865888  464.000038
#   466.132774  468.264104  470.394036  472.522578  474.649736  476.775519
#   478.899935  481.022991  483.144695  485.265055  487.384077  489.501771
#   491.618144  493.733203  495.846956  497.959411  500.070576  502.180458
#   504.289065  506.396404  508.502484  510.607312  512.710896  514.813243
#   516.914361  519.014258  521.112942  523.21042   525.3067    527.40179
#   529.495697  531.588429  533.679994  535.7704    537.859653  539.947763
#   542.034736  544.120581  546.205304  548.288915  550.371419  552.452826
#   554.533143  556.612377  558.690537  560.767629  562.843662  564.918643
#   566.992581  569.065482  571.137355  573.208207  575.278045  577.346878
#   579.414714  581.481559  583.547423  585.612311  587.676233  589.739195
#   591.801206  593.862273  595.922404  597.981607  600.039888  602.097257
#   604.153721  606.209287  608.263963  610.317757  612.370676  614.422729
#   616.473922  618.524265  620.573763  622.622426  624.67026   626.717274
#   628.763475  630.80887   632.853469  634.897277  636.940304  638.982556
#   641.024042  643.064768  645.104743  647.143975  649.182471  651.220239
#   653.257287  655.293621  657.329251  659.364184  661.398426  663.431987
#   665.464874  667.497094  669.528656  671.559566  673.589833  675.619464
#   677.648468  679.676851  681.704621  683.731787  685.758355  687.784334
#   689.809731  691.834554  693.858811  695.882509  697.905656  699.92826
#   701.950329  703.971869  705.99289   708.013398  710.033401  712.052907
#   714.071924  716.090459  718.108521  720.126116  722.143253  724.159938
#   726.176181  728.191988  730.207368  732.222328  734.236875  736.251018
#   738.264764  740.27812   742.291096  744.303697  746.315932  748.327809
#   750.339336  752.350519  754.361367  756.371888  758.382088  760.391977
#   762.401561  764.410848  766.419847  768.428564  770.437007  772.445184
#   774.453104  776.460772  778.468198  780.475389  782.482352  784.489096
#   786.495627  788.501954  790.508085  792.514027  794.519787  796.525374
#   798.530796  800.536059  802.541172  804.546142  806.550977  808.555685
#   810.560273  812.56475   814.569122  816.573398  818.577585  820.581691
#   822.585723  824.58969   826.593599  828.597458  830.601274  832.605056
#   834.60881   836.612545  838.616268  840.619987  842.62371   844.627444
#   846.631197  848.634977  850.638792  852.642649  854.646555  856.65052
#   858.654549  860.658652  862.662835  864.667106  866.671474  868.675946
#   870.680529  872.685231  874.690061  876.695024  878.700131  880.705387
#   882.710801  884.71638   886.722133  888.728067  890.734189  892.740507
#   894.747029  896.753763  898.760716  900.767897  902.775312  904.782969
#   906.790877  908.799043  910.807474  912.816178  914.825164  916.834438
#   918.844009  920.853883  922.86407   924.874576  926.885409  928.896577
#   930.908088  932.919949  934.932168  936.944753  938.957711  940.971051
#   942.984779  944.998904  947.013433  949.028374  951.043735  953.059522
#   955.075746  957.092411  959.109528  961.127102  963.145142  965.163656
#   967.182651  969.202135  971.222116  973.242601  975.263599  977.285116
#   979.30716   981.32974   983.352862  985.376536  987.400767  989.425565
#   991.450936  993.476889  995.503431  997.530569  999.558312 1001.586668]
# 已儲存為 CARS_selected_bands.csv
# 已儲存圖檔 CARS_Selected_Bands.png
#  LASSO 選到 9 個波段
# 波段索引: 15, 實際波長: 433.99 nm, LASSO 係數: 0.055954
# 波段索引: 19, 實際波長: 442.59 nm, LASSO 係數: 0.065812
# 波段索引: 55, 實際波長: 519.01 nm, LASSO 係數: -0.387041
# 波段索引: 116, 實際波長: 645.10 nm, LASSO 係數: 0.146909
# 波段索引: 148, 實際波長: 710.03 nm, LASSO 係數: 0.116759
# 波段索引: 153, 實際波長: 720.13 nm, LASSO 係數: 0.010885
# 波段索引: 154, 實際波長: 722.14 nm, LASSO 係數: 0.029013
# 波段索引: 200, 實際波長: 814.57 nm, LASSO 係數: -0.275674
# 波段索引: 292, 實際波長: 999.56 nm, LASSO 係數: 0.282250
# 已儲存為 LASSO_selected_bands.csv
# 已儲存圖檔 LASSO_Selected_Bands.png

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

# 設定字型
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# === 讀取資料 ===
healthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_CRAS/healthy")
unhealthy_dir = Path(r"C:/Users/Amanda/PycharmProjects/test/test_0415_LDA/npy_CRAS/unhealthy")

X = []
y = []
file_names = []

for f in healthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append(0)
    file_names.append(f.name)

for f in unhealthy_dir.glob("*.npy"):
    X.append(np.load(f))
    y.append(1)
    file_names.append(f.name)

X = np.array(X)
y = np.array(y)

wavelengths = np.load("wavelengths_remove_bands_300.npy")

n_iterations = 100
pls_components = 2
initial_bands = X.shape[1]

remaining_idx = np.arange(initial_bands)
best_score = -np.inf
best_subset = remaining_idx

print(f"開始 CARS, 初始 {initial_bands} 個波段")

for iter in range(n_iterations):
    print(f"\n迭代 {iter+1}/{n_iterations}，剩餘 {len(remaining_idx)} 波段")
    X_subset = X[:, remaining_idx]

    pls = PLSRegression(n_components=pls_components)
    pls.fit(X_subset, y)

    scores = cross_val_score(pls, X_subset, y, cv=5, scoring='r2')
    avg_score = np.mean(scores)
    print(f"   - CV R^2: {avg_score:.4f}")

    if avg_score > best_score:
        best_score = avg_score
        best_subset = remaining_idx.copy()
        print("[更新最佳波段組合]")

    coef = np.abs(pls.coef_[:, 0])
    sorted_idx = np.argsort(coef)
    n_remove = max(1, int(0.1 * len(remaining_idx)))
    remove_idx = sorted_idx[:n_remove]
    remaining_idx = np.delete(remaining_idx, remove_idx)

    if len(remaining_idx) <= pls_components:
        print("[波段數過少，自動停止]")
        break

# ======
print("\n=== CARS 結果 ===")
print(f"最佳 CV R^2: {best_score:.4f}")
print(f"選到的波段索引: {best_subset}")
print(f"選到的實際波長: {wavelengths[best_subset]}")

result_df = pd.DataFrame({
    "Band Index": best_subset,
    "Wavelength (nm)": wavelengths[best_subset]
})
result_df.to_csv("CARS_selected_bands.csv", index=False)
print("已儲存為 CARS_selected_bands.csv")

plt.figure(figsize=(12, 6))
plt.plot(wavelengths, np.zeros_like(wavelengths), 'k.', alpha=0.3)
plt.scatter(wavelengths[best_subset], np.zeros_like(best_subset), color='red', label='Selected Bands')
plt.xlabel("Wavelength (nm)")
plt.title("CARS Selected Bands")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("CARS_Selected_Bands.png", dpi=300)
print("已儲存圖檔 CARS_Selected_Bands.png")
plt.show()


# === CARS 結束 ===

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


X_cars = X[:, best_subset]

# 對 X_cars 做標準化（mean=0, std=1）
scaler = StandardScaler()
X_cars_scaled = scaler.fit_transform(X_cars)

# lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
# lasso.fit(X_cars_scaled, y)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_cars_scaled, y)

coef = lasso.coef_
selected_idx = np.where(coef != 0)[0]

final_selected_bands = best_subset[selected_idx]

print(f"\n LASSO 選到 {len(final_selected_bands)} 個波段")
for idx in final_selected_bands:
    print(f"波段索引: {idx}, 實際波長: {wavelengths[idx]:.2f} nm, LASSO 係數: {lasso.coef_[list(best_subset).index(idx)]:.6f}")

result_df = pd.DataFrame({
    "Band Index": final_selected_bands,
    "Wavelength (nm)": wavelengths[final_selected_bands],
    "LASSO Coef": lasso.coef_[selected_idx]
})
result_df.to_csv("LASSO_selected_bands.csv", index=False)
print("已儲存為 LASSO_selected_bands.csv")

plt.figure(figsize=(12, 6))
plt.stem(wavelengths[best_subset], coef)
plt.xlabel("Wavelength (nm)")
plt.ylabel("LASSO Coefficient")
plt.title("LASSO Coefficients for CARS-Selected Bands")
plt.grid(True)
plt.tight_layout()
plt.savefig("LASSO_Selected_Bands.png", dpi=300)
print("已儲存圖檔 LASSO_Selected_Bands.png")
plt.show()
