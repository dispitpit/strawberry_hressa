一次跑：
1. concentration = low / high
2. batch size = 8 / 16 / 32 / 64
3. experiment = no_aug / aug
總共：2 × 4 × 2 = 16 組實驗

輸出：
- ACC(Test)
- ACC(CV5)
- SP
- SN
- MCC

結果目錄結構：
D:\Amanda_strawberry\results\grid_compare_時間戳
├─ grid_compare_summary.csv
├─ grid_compare_summary_compact.csv
├─ low
│  ├─ bs_8
│  ├─ bs_16
│  ├─ bs_32
│  └─ bs_64
└─ high
   ├─ bs_8
   ├─ bs_16
   ├─ bs_32
   └─ bs_64

路徑設定：
- 真實資料：D:\Amanda_strawberry\2D_crop\low / high
    split CSV -> D:\Amanda_strawberry\ks\ks_low_divided_result / ks_high_divided_result
- fake 資料：
    low  -> D:\Amanda_strawberry\CycleGAN\final_result\final_use_low
    high -> D:\Amanda_strawberry\CycleGAN\final_result\final_use_high
