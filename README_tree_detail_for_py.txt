├─ 1.segmentation：影像分割與遮罩建立
│  │
│  ├─ Load_all_HIS.py：讀取 .hdr 與 .raw 高光譜資料，將每個樣本輸出為 300 張單波段 PNG
│  ├─ load_RGB.py：由高光譜資料建立虛擬 RGB 圖，供 U²-Net 分割訓練與檢查使用
│  ├─ binary_mask.py：將 U²-Net 輸出的灰階遮罩轉換為 0／255 二值遮罩
│  ├─ cleaned.py：對二值遮罩進行細節修正
│  ├─ test_mask_overlay_result.py：將遮罩疊加至虛擬 RGB 圖，視覺化檢查分割結果
│  ├─ all_image_mask.py：將單一植株遮罩套用至同一樣本的全部 300 個波段影像
│  │
│  ├─ U-2-Net-master
│  │  ├─ u2net_test.py：載入訓練完成的 U²-Net，對虛擬 RGB 圖產生植株遮罩
│  │  ├─ u2net_train.py：訓練 U²-Net 植株分割模型
│  │  ├─ data_loader.py：讀取 U²-Net 訓練所需的影像與遮罩
│  │  ├─ model
│  │  │  └─ u2net.py：U²-Net 模型架構定義
│  │  ├─ saved_models
│  │  │  └─ u2net
│  │  │     └─ u2net.pth：訓練完成的植株分割模型權重
│  │  └─ test_data
│  │     ├─ test_images：存放待分割的虛擬 RGB 圖 (模型用input)
│  │     └─ u2net_results：存放 U²-Net 輸出的植株遮罩 (模型用output)
│  │
│  └─ evaluate
│     ├─ mIOU.py：比較預測遮罩與人工標註遮罩，計算 IoU／mIoU 分割效能
│     ├─ change_file_name.py：調整檔名，使預測遮罩可與真實遮罩正確配對
│     └─ use_ corresponding_file.py：依檔名複製或篩選與真實標註相對應的 U²-Net 結果
│
├─ 2.analysis：高光譜反射率、vis與KS分析
│  │
│  ├─ single_data_test
│  │  └─ test.py：以單一 .hdr、.raw 與遮罩測試反射率擷取流程，輸出平均光譜 .npy 與 .csv
│  │
│  ├─ get_average_pixel_value.py：批次計算遮罩內植株像素的各波段平均反射率，建立全部樣本的 .npy 與 .csv
│  ├─ divide_healthy_unhealthy.py：依標籤將樣本區分為 healthy 與 unhealthy
│  ├─ dvide_low_high_vis.py：依接種濃度區分高濃度與低濃度資料
│  ├─ divide_low_high.py：區分高濃度與低濃度資料
│  ├─ save_wavelengths.py：從高光譜資料取得波段索引與實際波長的對應關係
│  ├─ wavelengths.npy：儲存 300 個波段對應的實際波長，供植生指數、特徵選擇與模型程式使用(重要input)
│  │
│  ├─ single_potted_plant_reflectance
│  │  └─ S6_mean_picture.py：彙整單一盆栽不同日期的平均反射率，時序光譜曲線 (S6為例)
│  │
│  └─ vis_count
│     ├─ vissss.py：根據平均反射率計算植生指數，輸出植生指數表與波段匹配誤差
│     ├─ RF.py：以 ReliefF 評估各植生指數的重要性(主)
│     ├─ K_sensitivity.py：比較 ReliefF 在不同鄰居數 K 下的重要性結果(測試調參用)
│     ├─ K_mean.py：整合選定 K 值的 ReliefF 結果，取得較穩定的重要植生指數
│     ├─ ReliefF_source_code.py：ReliefF 的原始碼(拿來看的)
│     │
│     └─ KS_v2
│        ├─ divided_high_low_vis_v2.py：將完整植生指數表分為高濃度與低濃度資料
│        ├─ KS_high_v2.py：對高濃度樣本執行 Kennard–Stone 資料切分(重要input)
│        ├─ KS_low_v2.py：對低濃度樣本執行 Kennard–Stone 資料切分(重要input)
│        └─ KS_npy_v2.py：對全部濃度或平均光譜資料執行 Kennard–Stone 資料切分(重要input)
│
├─ 3.feature selection：離群值檢查與重要波段選擇
│  │
│  ├─ create_all_train_dataset.py：彙整並轉移特徵選擇所需的全部訓練資料
│  ├─ Normal_check_RT.py：檢查各波段反射率資料是否符合常態分布
│  ├─ Find_outlier_sample.py：找出光譜離群樣本並輸出離群值清單
│  ├─ outlier_samples.csv：記錄被判定為離群值的樣本
│  │
│  └─ feature_select
│     ├─ CARS.py：使用競爭性自適應重加權取樣選擇重要波段
│     ├─ LDA.py：利用線性判別分析評估可區分健康與感染樣本的波段
│     ├─ PCA.py：利用主成分分析進行光譜降維與特徵分析
│     ├─ RF.py：使用隨機森林特徵重要性選擇重要波段
│     └─ SPA.py：使用連續投影演算法降低波段共線性並選擇代表波段
│
├─ 4.ML：以植生指數、完整光譜及篩選波段進行機器學習分類
│  │
│  ├─ vis_count
│  │  │
│  │  ├─ VIs_base_v2 (主)
│  │  │  ├─ vis_base_high_v2.py：使用高濃度植生指數訓練與評估多種機器學習模型
│  │  │  ├─ vis_base_low_v2.py：使用低濃度植生指數訓練與評估多種機器學習模型
│  │  │  └─ vis_base_npy_v2.py：使用全部濃度植生指數訓練與評估多種機器學習模型
│  │  │
│  │  └─ KS_v2 (主)
│  │     ├─ KS_high_v2.py：建立高濃度樣本的 KS 訓練集與測試集
│  │     ├─ KS_low_v2.py：建立低濃度樣本的 KS 訓練集與測試集
│  │     ├─ KS_npy_v2.py：建立全部濃度平均光譜的 KS 訓練集與測試集
│  │     ├─ ML_v2.py：使用完整 300 波段資料訓練與比較多種機器學習模型
│  │     └─ ML_v3_feature_selection.py：使用篩選後的重要波段訓練與比較機器學習模型
│  │
│  └─ KS_reload (副)
│     ├─ ML.py：舊版單一輸入設定的機器學習訓練程式
│     ├─ predict_one.py：載入已儲存模型，對單一樣本進行健康／感染預測
│     └─ model.joblib：儲存訓練完成的機器學習模型
│
├─ 5.CycleGAN+DL：生成式資料增強與 2D／3D 深度學習
│  │
│  ├─ ks
│  │  └─ generate_ks_lists.py：依植生指數特徵對健康與感染樣本分別執行 KS 切分，建立 train、val、test 路徑清單
│  │
│  ├─ pack_wavelength_png_to_npy.py：將同一樣本的 300 張單波段 PNG 依波長排序並堆疊為 (300, H, W) 高光譜立方體
│  │
│  ├─ 2D_crop
│  │  └─ preprocess_crop_plant.py：以最大值投影、二值化與形態學處理定位植株，裁切主要植株區域並減少背景
│  │
│  ├─ all_taining_data.py：載入訓練完成的 CycleGAN，以重疊 patch 生成 32 波段假資料，輸出 .npy 與 RGB 預覽圖
│  ├─ true_vs_fake_v2.py：計算真實與生成樣本的平均反射率，繪製 32 波段光譜曲線進行比較(input需先u2-net)
│  │
│  ├─ CycleGAN
│  │  │
│  │  ├─ cycleGAN.zip：CycleGAN 相關設定壓縮備份(解壓到當前資料夾)
│  │  ├─ pytorch-CycleGAN-and-pix2pix-master.zip：PyTorch CycleGAN／pix2pix 原始程式碼壓縮檔
│  │  │
│  │  ├─ checkpoints(紀錄)
│  │  │
│  │  ├─ datasets
│  │  │  ├─ combine_A_and_B.py：將 A、B 兩個資料域的影像進行組合，建立模型所需的資料格式
│  │  │  ├─ make_dataset_aligned.py：整理並建立對齊式 A／B 影像資料集
│  │  │  └─ prepare_cityscapes_dataset.py：原始框架提供的 Cityscapes 資料集前處理工具
│  │  │
│  │  ├─ final_result
│  │  │  ├─ all_taining_data_v1.py
│  │  │  ├─ all_training_data_v3.py：批次生成產生後續深度學習使用之 CycleGAN 擴增資料
│  │  │  ├─ one_sample_check.py：針對單一樣本檢查 CycleGAN 生成結果與資料格式
│  │  │  ├─ true_vs_fake_v1.py：比較真實與生成樣本之光譜反射率與生成結果(主)
│  │  │  └─ true_vs_fake_v7_more_loss.py：進一步比較真實與生成樣本(測試用)
│  │  │
│  │  └─ test_models
│  │     ├─ check_file.py：檢查輸入資料、檔案路徑與生成結果是否完整
│  │     ├─ infer_full_hsi_tiles.py：將完整高光譜影像切成 tiles／patches 進行 CycleGAN 推論並輸出生成結果
│  │     └─ real_vs_fake_line.py：繪製真實與生成樣本之光譜曲線，用於比較兩者反射率走向(副)
│  │
│  ├─ no_aug：未加入 CycleGAN 假資料的基準深度學習實驗(副)
│  │  │
│  │  ├─ 2D_v2
│  │  │  ├─ config_2d.py：設定 2D 模型的資料路徑、超參數、輸入波段與訓練選項(調參)
│  │  │  ├─ dataset_hsi_2d_patch.py：讀取高光譜 .npy、標籤與影像 patch，建立 2D 模型輸入
│  │  │  ├─ model_2d.py：定義一般 2D CNN 模型架構(測試用的)
│  │  │  ├─ model_2d_resnet18.py：定義以 ResNet18 為骨幹的 2D 分類模型
│  │  │  └─ train_2d.py：建立 DataLoader、訓練及評估未增強的 2D 模型
│  │  │
│  │  └─ 3D
│  │     ├─ config_3d.py：設定 3D 模型的資料路徑、超參數與輸入深度(調參)
│  │     ├─ dataset_hsi_3d_patch.py：讀取高光譜立方體並擷取 3D patch
│  │     ├─ model_3d.py：定義一般 3D CNN 模型架構(測試用的)
│  │     ├─ model_3d_resnet18.py：定義 3D ResNet18 分類模型
│  │     └─ train_3d.py：訓練及評估未加入生成資料的 3D 模型
│  │
│  └─ all_aug_and_no_aug：比較加入與未加入 CycleGAN 假資料的分類結果(主)
│     │
│     ├─ CNN_2D
│     │  ├─ config_2d.py：設定 2D 增強實驗的資料來源、超參數與選定波段(調參)
│     │  ├─ dataset_hsi_2d_patch.py：同時讀取真實與生成高光譜資料，建立 2D patch 輸入
│     │  ├─ model_2d.py：定義 2D CNN 模型(測試用的)
│     │  ├─ model_2d_resnet18.py：定義 ResNet18 形式的 2D 分類模型
│     │  ├─ train_2d.py：分別訓練加入與未加入 CycleGAN 資料的 2D 模型並比較結果
│     │  └─ test.py：載入訓練完成的 2D 模型進行獨立測試
│     │
│     └─ CNN_3D
│        ├─ config_3d.py：設定 3D 增強實驗的資料來源與訓練參數(調參)
│        ├─ dataset_hsi_3d.py：讀取真實與生成的高光譜立方體，建立 3D 模型輸入
│        ├─ model_3d.py：定義 3D CNN 模型(測試用的)
│        ├─ model_3d_resnet18.py：定義 3D ResNet18 分類模型
│        └─ train_3d.py：分別訓練加入與未加入 CycleGAN 資料的 3D 模型並比較結果
