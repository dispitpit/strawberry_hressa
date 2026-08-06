from spectral import *
import numpy as np

hdr_file = r"C:\Users\Amanda\PycharmProjects\test\病害高光譜2024June\20240620\sample1.hdr"

img = open_image(hdr_file)
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])

np.save("wavelengths.npy", wavelengths)
print(f"已儲存 wavelengths.npy，總波段數量：{len(wavelengths)}")
print("finish")