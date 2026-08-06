import numpy as np
import pandas as pd
from pathlib import Path

def npy_to_excel(npy_path, xlsx_path=None):
    npy_path = Path(npy_path)
    if xlsx_path is None:
        xlsx_path = npy_path.with_suffix(".xlsx")

    data = np.load(npy_path, allow_pickle=True)

    def to_df(data):
        # case A: dict-like
        if isinstance(data, dict):
            keys = {k.lower(): k for k in data.keys()}

            wl_key = next((keys[k] for k in ["wavelengths", "wavelength", "wl"]), None)
            idx_key = next((keys[k] for k in ["index", "indices", "idx"]), None)

            if wl_key is not None and idx_key is not None:
                return pd.DataFrame({"Index": data[idx_key], "Wavelength (nm)": data[wl_key]})
            elif wl_key is not None:
                return pd.DataFrame({"Index": range(len(data[wl_key])), "Wavelength (nm)": data[wl_key]})
            else:
                return pd.DataFrame(data)

        # case B: 結構化陣列
        if isinstance(data, np.ndarray) and data.dtype.names:
            names = [n.lower() for n in data.dtype.names]
            cols = {}
            idx_field = next((n for n in data.dtype.names if n.lower() in ["index","indices","idx"]), None)
            wl_field  = next((n for n in data.dtype.names if n.lower() in ["wavelengths","wavelength","wl"]), None)
            if idx_field or wl_field:
                if idx_field: cols["Index"] = data[idx_field]
                if wl_field:  cols["Wavelength (nm)"] = data[wl_field]
                return pd.DataFrame(cols)
            return pd.DataFrame({name: data[name] for name in data.dtype.names})

        # case C: 一般 ndarray
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({"Index": np.arange(len(data)), "Wavelength (nm)": data})
            if data.ndim == 2:
                if data.shape[1] == 2:
                    a, b = data[:,0], data[:,1]

                    def looks_like_index(x):
                        x_int = np.allclose(x, np.round(x))
                        monotonic = np.all(np.diff(x) >= 0)
                        return x_int and monotonic
                    if looks_like_index(a) and not looks_like_index(b):
                        return pd.DataFrame({"Index": a.astype(int), "Wavelength (nm)": b})
                    elif looks_like_index(b) and not looks_like_index(a):
                        return pd.DataFrame({"Index": b.astype(int), "Wavelength (nm)": a})
                    else:
                        return pd.DataFrame({"Col0": a, "Col1": b})
                else:
                    cols = {f"Col{i}": data[:, i] for i in range(data.shape[1])}
                    return pd.DataFrame(cols)

        # 其他型別
        try:
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"無法轉成表格：{type(data)}; 錯誤：{e}")

    df = to_df(data)

    # 整理一下
    if "Wavelength (nm)" in df.columns:
        try:
            df["Wavelength (nm)"] = pd.to_numeric(df["Wavelength (nm)"], errors="coerce")

            df["Wavelength (nm)"] = df["Wavelength (nm)"].round(10)
        except Exception:
            pass

    # 若有 Index 欄，轉成 int
    if "Index" in df.columns:
        try:
            df["Index"] = pd.to_numeric(df["Index"], errors="coerce").astype("Int64")
        except Exception:
            pass

    # 輸出
    df.to_excel(xlsx_path, index=False)
    return xlsx_path, df.shape

if __name__ == "__main__":
    out_path, shape = npy_to_excel("wavelengths.npy")
    # print(f"已輸出：{out_path}，表格大小：{shape}")
