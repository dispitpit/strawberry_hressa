執行方式:
python train_3d.py


2D 與 3D 定義差別
-----------------------
1) 2D CNN (band 是 channel)
   - 輸入 shape: (B, C, H, W)
   - 其中 C = selected bands 數量，例如 32
   - 也就是把每個 spectral band 當成 Conv2d 的 input channel
   - Conv2d kernel 只在空間平面 (H, W) 上滑動
   - 光譜資訊會透過 channel mixing 被使用，但不把 band 順序當成可滑動的 depth 軸

2) 3D CNN (band 是 depth)
   - 輸入 shape: (B, 1, D, H, W)
   - 其中 D = selected bands 數量，例如 32
   - 也就是把 spectral bands 視為 depth 軸，交給 Conv3d 做 (D, H, W) 聯合建模
   - Conv3d kernel 會同時在光譜與空間維度滑動
   - 更適合表達局部 spectral-spatial pattern
-----------------------
