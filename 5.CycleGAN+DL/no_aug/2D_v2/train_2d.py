# train_2d.py — All-in-One Version
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    matthews_corrcoef
)

# record log
from pathlib import Path
from datetime import datetime
import sys

from config_2d import *
# 全尺寸版本
# from dataset_2d import HSIDataset2D
# from model_2d import SimpleHSI2DNet
# =======================================
from model_2d_resnet18 import HSIResNet18
MODEL_TAG = "HSIResNet18"
#========================================
from dataset_hsi_2d_patch import HSIDataset2D_Patch


# ---------------------------------------------------------
#  draw learning curves
# ---------------------------------------------------------
def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path="curve.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------
#  Train loop
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc=f"[Train {epoch}/{total_epochs}]", colour="green")

    for x, y, _ in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        loop.set_postfix(loss=total_loss/total, acc=correct/total)

    return total_loss / total, correct / total

# ---------------------------------------------------------
#  Validation loop
# ---------------------------------------------------------
def eval_one_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc=f"[Val {epoch}]", colour="cyan")
    with torch.no_grad():
        for x, y, _ in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            loop.set_postfix(loss=total_loss/total, acc=correct/total)

    return total_loss / total, correct / total


# ---------------------------------------------------------
#  Test function
# ---------------------------------------------------------
def evaluate_test(model, loader, device):
    model.eval()

    # --------- patch-level ---------
    y_true_patch, y_pred_patch, y_prob_patch = [], [], []

    # --------- image-level  ---------
    image_probs = {}
    image_labels = {}

    with torch.no_grad():
        for x, y, img_idx in loader:
            x = x.to(device)
            logits = model(x)

            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(1).cpu().numpy()
            y_np = y.numpy()
            img_idx_np = img_idx.numpy()

            # --- patch-level ---
            y_true_patch.extend(y_np)
            y_pred_patch.extend(pred)
            y_prob_patch.extend(prob.cpu().numpy())

            # --- image-level ---
            for lbl, p, iid in zip(y_np, prob.cpu().numpy(), img_idx_np):
                if iid not in image_probs:
                    image_probs[iid] = []
                    image_labels[iid] = int(lbl)
                image_probs[iid].append(float(p))


    # ===== image-level 指標 =====
    img_ids = sorted(image_labels.keys())
    y_true_img = np.array([image_labels[i] for i in img_ids])
    y_prob_img = np.array([np.mean(image_probs[i]) for i in img_ids])
    y_pred_img = (y_prob_img >= 0.5).astype(int)


    cm_img = confusion_matrix(y_true_img, y_pred_img)
    tn, fp, fn, tp = cm_img.ravel()
    sn_img = tp / (tp + fn + 1e-6)
    sp_img = tn / (tn + fp + 1e-6)
    acc_img = (tp + tn) / (tp + tn + fp + fn)
    mcc_img = matthews_corrcoef(y_true_img, y_pred_img)

    if np.unique(y_true_img).size < 2:
        print("[WARN] Image-level y_true 只有單一類別，AUC 無法定義，設為 NaN")
        auc_img = float("nan")
    else:
        auc_img = roc_auc_score(y_true_img, y_prob_img)

    print("\n====== IMAGE-LEVEL TEST RESULTS ======")
    print("Confusion Matrix (image):\n", cm_img)
    print("ACC(image):", acc_img)
    print("SP (image):", sp_img)
    print("SN (image):", sn_img)
    print("MCC(image):", mcc_img)
    print("AUC(image):", auc_img)
    print("\nClassification Report (image):\n")
    print(classification_report(y_true_img, y_pred_img))

    # ===== patch-level 指標 =====
    y_true_patch = np.array(y_true_patch)
    y_pred_patch = np.array(y_pred_patch)
    y_prob_patch = np.array(y_prob_patch)

    cm_patch = confusion_matrix(y_true_patch, y_pred_patch)
    tn, fp, fn, tp = cm_patch.ravel()
    sn_patch = tp / (tp + fn + 1e-6)
    sp_patch = tn / (tn + fp + 1e-6)
    acc_patch = (tp + tn) / (tp + tn + fp + fn)
    mcc_patch = matthews_corrcoef(y_true_patch, y_pred_patch)
    if np.unique(y_true_patch).size < 2:
        print("[WARN] Patch-level y_true 只有單一類別，AUC 無法定義，設為 NaN")
        auc_patch = float("nan")
    else:
        auc_patch = roc_auc_score(y_true_patch, y_prob_patch)

    print("\n====== PATCH-LEVEL TEST RESULTS (for reference) ======")
    print("Confusion Matrix (patch):\n", cm_patch)
    print("SN (patch):", sn_patch)
    print("SP (patch):", sp_patch)
    print("ACC(patch):", acc_patch)
    print("MCC(patch):", mcc_patch)
    print("AUC(patch):", auc_patch)

    return sn_img, sp_img, acc_img, mcc_img, auc_img, cm_img

def setup_experiment_dir(model_tag: str):
    """
    建立本次實驗的結果資料夾，例如：
      results/high_patch128_20251121_210512/
        logs/, ckpt/, figures/, info/
    """
    base = Path("results")
    exp_name = f"{CONCENTRATION}_patch128_{model_tag}_{datetime.now():%Y%m%d_%H%M%S}"
    exp_dir = base / exp_name

    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "ckpt").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    (exp_dir / "info").mkdir(exist_ok=True)

    return exp_dir


class Tee:
    """
    簡單的 stdout tee：
    - 所有 print 仍顯示在 console
    - 同步寫入指定 log 檔
    """
    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self.log_file.write(data)

    def flush(self):
        self._stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self._stdout
        self.log_file.close()


# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------
def main():
    # ====  log ====
    exp_dir = setup_experiment_dir(MODEL_TAG)
    tee = Tee(exp_dir / "logs" / "train_log.txt")

    # 存
    if SELECTED_BANDS is not None:
        sb_path = exp_dir / "info" / "selected_bands.npy"
        np.save(sb_path, np.array(SELECTED_BANDS, dtype=int))
        print(f"[INFO] Saved selected band indices to {sb_path}")

    # 再存一份
    config_txt = exp_dir / "info" / "config_snapshot.txt"
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write(f"CONCENTRATION = {CONCENTRATION}\n")
        f.write(f"TRAIN_CSV     = {TRAIN_CSV}\n")
        f.write(f"TEST_CSV      = {TEST_CSV}\n")
        f.write(f"BASE_DATA     = {BASE_DATA}\n")
        f.write(f"BATCH_SIZE    = {BATCH_SIZE}\n")
        f.write(f"LR            = {LR}\n")
        f.write(f"EPOCHS        = {EPOCHS}\n")
        f.write(f"IN_CHANNELS   = {IN_CHANNELS}\n")
        f.write(f"NUM_CLASSES   = {NUM_CLASSES}\n")
        f.write(f"DEVICE        = {DEVICE}\n")
    print(f"[INFO] Saved config snapshot to {config_txt}")


    # 開始!!!!
    try:
        print(f"[INFO] Experiment directory: {exp_dir}")
        print("\n==========================================")
        print("      [Stage 1] Loading PATCH Dataset     ")
        print("==========================================")

        full_train_ds = HSIDataset2D_Patch(
            split_csv=TRAIN_CSV,
            data_root=BASE_DATA,
            selected_bands=SELECTED_BANDS
        )

        test_ds = HSIDataset2D_Patch(
            split_csv=TEST_CSV,
            data_root=BASE_DATA,
            selected_bands=SELECTED_BANDS
        )

        print(f"[INFO] Total training patches : {len(full_train_ds)}")
        print(f"[INFO] Total testing  patches : {len(test_ds)}\n")

        # ------------------------------------------------------
        print("==========================================")
        print("   [Stage 2] Splitting Train / Val Sets   ")
        print("==========================================")

        n_total = len(full_train_ds)
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val

        train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

        print(f"[INFO] Train patches : {n_train}")
        print(f"[INFO] Val   patches : {n_val}\n")

        # ------------------------------------------------------
        print("==========================================")
        print("      [Stage 3] Creating DataLoader       ")
        print("==========================================")

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        print(f"[INFO] Batch size         : {BATCH_SIZE}")
        print(f"[INFO] Num workers        : 4")
        print(f"[INFO] Train steps/epoch  : {len(train_loader)}")
        print(f"[INFO] Val   steps/epoch  : {len(val_loader)}")
        print(f"[INFO] Test  steps/epoch  : {len(test_loader)}\n")

        # ------------------------------------------------------
        print("==========================================")
        print("          [Stage 4] Building Model        ")
        print("==========================================")

        # from model_2d import SimpleHSI2DNet
        # model = SimpleHSI2DNet(IN_CHANNELS, NUM_CLASSES).to(DEVICE)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=LR)

        # from model_2d_resnet18 import HSIResNet18
        model = HSIResNet18(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            pretrained=True,  # 如不想用預訓練可改 False
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)


        print(model)
        print(f"\n[INFO] Learning rate = {LR}")
        print(f"[INFO] Device = {DEVICE}\n")

        # ------------------------------------------------------
        print("==========================================")
        print("         [Stage 5] Start Training         ")
        print("==========================================")

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0

        for epoch in range(1, EPOCHS + 1):

            print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

            # ---- Training ----
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS
            )

            # ---- Validation ----
            va_loss, va_acc = eval_one_epoch(
                model, val_loader, criterion, DEVICE, epoch
            )

            train_losses.append(tr_loss)
            val_losses.append(va_loss)
            train_accs.append(tr_acc)
            val_accs.append(va_acc)

            print(f"[Summary] Train Acc={tr_acc:.4f}, Val Acc={va_acc:.4f}")

            # ---- Save Best Model ----
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                ckpt_path = exp_dir / "ckpt" / "best_model.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"[INFO] New best model saved to {ckpt_path}")

        # ------------------------------------------------------
        print("\n==========================================")
        print("      [Stage 6] Plot Learning Curves      ")
        print("==========================================")

        curve_path = exp_dir / "figures" / "training_curve.png"
        plot_curves(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            out_path=curve_path
        )
        print(f"[INFO] Saved training curves as {curve_path}")

        # ------------------------------------------------------
        print("\n==========================================")
        print("             [Stage 7] Testing            ")
        print("==========================================")

        best_ckpt = exp_dir / "ckpt" / "best_model.pth"
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        sn, sp, acc, mcc, auc, cm = evaluate_test(model, test_loader, DEVICE)

        # ---- 存 metrics  ----
        metrics_path = exp_dir / "info" / "metrics_image_level.txt"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("SN\tSP\tACC\tMCC\tAUC\n")
            f.write(f"{sn:.6f}\t{sp:.6f}\t{acc:.6f}\t{mcc:.6f}\t{auc:.6f}\n")
        print(f"[INFO] Saved metrics to {metrics_path}")

        # ---- 畫 confusion matrix ----
        fig_cm = exp_dir / "figures" / "confusion_matrix_image.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Image-level)")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Healthy", "Unhealthy"])
        plt.yticks(tick_marks, ["Healthy", "Unhealthy"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(fig_cm, dpi=200)
        plt.close()
        print(f"[INFO] Saved confusion matrix figure to {fig_cm}")

        print("\n============== Done ==============")

    finally:
        tee.close()

if __name__ == "__main__":
    main()