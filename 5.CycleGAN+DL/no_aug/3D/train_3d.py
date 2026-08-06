# train_3d.py
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, matthews_corrcoef

from config_3d import *
from dataset_hsi_3d_patch import HSIDataset3D_Patch
# from model_3d import SimpleHSI3DNet
from model_3d_resnet18 import HSIResNet3D18


def setup_experiment_dir(model_name: str):
    base = Path("results")
    exp_name = f"{CONCENTRATION}_{model_name}_3D_ps{PATCH_SIZE}_{datetime.now():%Y%m%d_%H%M%S}"
    exp_dir = base / exp_name
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "ckpt").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    (exp_dir / "info").mkdir(exist_ok=True)
    return exp_dir

# log紀錄
class Tee:
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc=f"[Train {epoch}/{total_epochs}]")

    for x, y, _ in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        loop.set_postfix(loss=total_loss / max(total, 1), acc=correct / max(total, 1))

    return total_loss / max(total, 1), correct / max(total, 1)


def eval_one_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(loader, desc=f"[Val {epoch}]")
    with torch.no_grad():
        for x, y, _ in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            loop.set_postfix(loss=total_loss / max(total, 1), acc=correct / max(total, 1))

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_test_image_level(model, loader, device, thresh=0.5):
    model.eval()

    image_probs = {}
    image_labels = {}

    with torch.no_grad():
        for x, y, img_idx in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            y_np = y.numpy()
            img_idx_np = img_idx.numpy()

            for lbl, p, iid in zip(y_np, prob, img_idx_np):
                if iid not in image_probs:
                    image_probs[iid] = []
                    image_labels[iid] = int(lbl)
                image_probs[iid].append(float(p))

    img_ids = sorted(image_labels.keys())
    y_true = np.array([image_labels[i] for i in img_ids])
    y_prob = np.array([np.mean(image_probs[i]) for i in img_ids])
    y_pred = (y_prob >= thresh).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sn = tp / (tp + fn + 1e-6)
    sp = tn / (tn + fp + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    mcc = matthews_corrcoef(y_true, y_pred)


    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    print("\n====== IMAGE-LEVEL TEST RESULTS (3D) ======")
    print("Confusion Matrix:\n", cm)

    print("ACC:", acc)
    print("SN:", sn)
    print("SP:", sp)
    print("MCC:", mcc)
    print("AUC:", auc)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    return sn, sp, acc, mcc, auc, cm


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path: Path):
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


def main():
    model_name = "SimpleHSI3DNet"
    exp_dir = setup_experiment_dir(model_name=model_name)
    tee = Tee(exp_dir / "logs" / "train_log.txt")

    try:
        print(f"[INFO] Experiment dir: {exp_dir}")
        print(f"[INFO] Device: {DEVICE}")
        print(f"[INFO] Selected bands (D={SPECTRAL_DEPTH}): {SELECTED_BANDS}")

        with open(exp_dir / "info" / "config_snapshot.txt", "w", encoding="utf-8") as f:
            f.write(f"CONCENTRATION={CONCENTRATION}\n")
            f.write(f"BASE_DATA={BASE_DATA}\n")
            f.write(f"TRAIN_CSV={TRAIN_CSV}\n")
            f.write(f"TEST_CSV={TEST_CSV}\n")
            f.write(f"PATCH_SIZE={PATCH_SIZE}\n")
            f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
            f.write(f"LR={LR}\n")
            f.write(f"EPOCHS={EPOCHS}\n")
            f.write(f"NUM_WORKERS={NUM_WORKERS}\n")
            f.write(f"SPECTRAL_DEPTH={SPECTRAL_DEPTH}\n")
            f.write(f"SELECTED_BANDS={SELECTED_BANDS}\n")

        full_train_ds = HSIDataset3D_Patch(
            split_csv=TRAIN_CSV,
            data_root=BASE_DATA,
            patch_size=PATCH_SIZE,
            selected_bands=SELECTED_BANDS,
        )
        test_ds = HSIDataset3D_Patch(
            split_csv=TEST_CSV,
            data_root=BASE_DATA,
            patch_size=PATCH_SIZE,
            selected_bands=SELECTED_BANDS,
        )

        print(f"[INFO] Train patches total: {len(full_train_ds)}")
        print(f"[INFO] Test  patches total: {len(test_ds)}")

        # split train/val
        n_total = len(full_train_ds)
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_train_ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        print(f"[INFO] Steps/epoch train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

        # model
        # IN MODEL 3D
        # model = SimpleHSI3DNet(num_classes=NUM_CLASSES, base=16).to(DEVICE)
        model_name = "HSIResNet3D18"
        model = HSIResNet3D18(
            num_classes=NUM_CLASSES,
            in_channels=1,
            pretrained=False,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        print(model)

        # train
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = -1.0

        for epoch in range(1, EPOCHS + 1):
            print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS)
            va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, DEVICE, epoch)

            train_losses.append(tr_loss); val_losses.append(va_loss)
            train_accs.append(tr_acc);   val_accs.append(va_acc)

            print(f"[Summary] Train Acc={tr_acc:.4f}, Val Acc={va_acc:.4f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                ckpt_path = exp_dir / "ckpt" / "best_model.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"[INFO] Saved best model: {ckpt_path}")

        curve_path = exp_dir / "figures" / "training_curve.png"
        plot_curves(train_losses, val_losses, train_accs, val_accs, curve_path)
        print(f"[INFO] Saved curve: {curve_path}")

        best_ckpt = exp_dir / "ckpt" / "best_model.pth"
        state = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)

        sn, sp, acc, mcc, auc, cm = evaluate_test_image_level(model, test_loader, DEVICE)

        metrics_path = exp_dir / "info" / "metrics_image_level.txt"
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("SN\tSP\tACC\tMCC\tAUC\n")
            f.write(f"{sn:.6f}\t{sp:.6f}\t{acc:.6f}\t{mcc:.6f}\t{auc:.6f}\n")
        print(f"[INFO] Saved metrics: {metrics_path}")

        fig_cm = exp_dir / "figures" / "confusion_matrix_image.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Image-level, 3D)")
        plt.colorbar()
        tick = np.arange(2)
        plt.xticks(tick, ["Healthy", "Unhealthy"])
        plt.yticks(tick, ["Healthy", "Unhealthy"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(fig_cm, dpi=200)
        plt.close()
        print(f"[INFO] Saved CM fig: {fig_cm}")

        print("\n============== Done ==============")

    finally:
        tee.close()


if __name__ == "__main__":
    main()
