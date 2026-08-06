import numpy as np
from joblib import load

model = load("model.joblib")

x = np.load(r"D:\Users\Amanda\PycharmProjects\test\test_0322_new.raw_pca(wait)\npy\D0703_S4_mean_spectrum.npy")

# 如果是 1 維資料，要變成 (1, 特徵數)
if x.ndim == 1:
    x = x.reshape(1, -1)

y_pred = model.predict(x)
y_prob = model.predict_proba(x)

if y_pred == 0:
    print("健康植株")
else:
    print("預測到草莓炭疽病:")
    print(f"病症機率為{y_prob[1]}")


