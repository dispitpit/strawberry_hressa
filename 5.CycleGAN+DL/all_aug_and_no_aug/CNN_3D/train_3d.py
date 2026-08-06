# train_3d.py
# 3D HSI classification full pipeline
# 1. concentration = low / high
# 2. batch size sweep（預設 [4, 8, 16]）
# 3. no_aug / aug
# 4. 每組輸出 ACC(Test), ACC(CV5), SP, SN, MCC

import gc
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import config_3d as cfg
from dataset_hsi_3d import HSI3DDataset
from model_3d import SimpleHSI3DNet
from model_3d_resnet18 import HSIResNet3D18
from utils_records import build_real_records, build_aug_records_from_folders


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_master_result_root() -> Path:
    out_dir = cfg.RESULTS_ROOT / f"grid_compare_3d_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def print_runtime_info():
    print("=" * 100)
    print("[Runtime-3D]")
    print("cfg.DEVICE                =", cfg.DEVICE)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
        print("AMP enabled              =", bool(cfg.USE_AMP))
    else:
        print("AMP enabled              = False (CUDA unavailable)")
    print("MODEL_NAME               =", cfg.MODEL_NAME)
    print("PIN_MEMORY               =", cfg.PIN_MEMORY)
    print("NUM_WORKERS              =", cfg.NUM_WORKERS)
    print("TARGET_HW                =", cfg.TARGET_HW)
    print("SPECTRAL_DEPTH           =", cfg.SPECTRAL_DEPTH)
    print("CACHE_ROOT               =", cfg.CACHE_ROOT)
    print("CONCENTRATIONS           =", cfg.CONCENTRATIONS)
    print("BATCH_SIZES              =", cfg.BATCH_SIZES)
    print("=" * 100)


def resolve_concentration_env(concentration: str):
    if concentration not in cfg.CONCENTRATIONS:
        raise ValueError(f"Unknown concentration: {concentration}")

    train_csv = cfg.BASE_KS / f"ks_{concentration}_divided_result" / f"ks_split_train_{concentration}.csv"
    test_csv = cfg.BASE_KS / f"ks_{concentration}_divided_result" / f"ks_split_test_{concentration}.csv"
    base_data = cfg.BASE_DATA_ROOT / concentration

    aug_dir_name = cfg.AUG_DIR_MAP[concentration]
    aug_root = cfg.AUG_BASE_ROOT / aug_dir_name
    aug_csv = aug_root / "aug_manifest.csv"

    return {
        "concentration": concentration,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "base_data": base_data,
        "aug_root": aug_root,
        "aug_csv": aug_csv,
    }


def make_loader(dataset, batch_size, shuffle):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    if cfg.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def make_model():
    if cfg.MODEL_NAME == "simple3d":
        model = SimpleHSI3DNet(in_channels=cfg.INPUT_CHANNELS_3D, num_classes=cfg.NUM_CLASSES)
    elif cfg.MODEL_NAME == "resnet18_3d":
        model = HSIResNet3D18(
            in_channels=cfg.INPUT_CHANNELS_3D,
            num_classes=cfg.NUM_CLASSES,
            pretrained=cfg.PRETRAINED,
        )
    else:
        raise ValueError(f"Unknown MODEL_NAME: {cfg.MODEL_NAME}")
    return model.to(cfg.DEVICE)


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=cfg.LR)


def make_criterion():
    return torch.nn.CrossEntropyLoss()


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    acc = (tp + tn) / total
    sp = tn / max(tn + fp, 1)
    sn = tp / max(tp + fn, 1)
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")

    return {
        "acc": float(acc),
        "sp": float(sp),
        "sn": float(sn),
        "mcc": float(mcc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled=False, debug_first_batch=False):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if debug_first_batch and batch_idx == 0:
            print(f"[DEBUG-3D] first train batch x.shape={tuple(x.shape)}, x.device={x.device}, y.device={y.device}")

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_n += x.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled=False, debug_first_batch=False):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if debug_first_batch and batch_idx == 0:
            print(f"[DEBUG-3D] first val/test batch x.shape={tuple(x.shape)}, x.device={x.device}, y.device={y.device}")

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_n += x.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


@torch.no_grad()
def collect_predictions(model, loader, device, amp_enabled=False):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    if len(all_probs) == 0:
        return {
            "y_true": np.array([], dtype=np.int64),
            "y_pred": np.array([], dtype=np.int64),
            "prob_pos": np.array([], dtype=np.float32),
        }

    return {
        "y_true": np.concatenate(all_targets).astype(np.int64),
        "y_pred": np.concatenate(all_preds).astype(np.int64),
        "prob_pos": np.concatenate(all_probs).astype(np.float32),
    }


def make_fold_train_df(fold_train_real: pd.DataFrame, aug_df: pd.DataFrame, use_aug: bool) -> pd.DataFrame:
    if not use_aug or len(aug_df) == 0:
        return fold_train_real.copy().reset_index(drop=True)

    train_origin_ids = set(fold_train_real["origin_id"].astype(str).tolist())
    fold_aug = aug_df[aug_df["origin_id"].astype(str).isin(train_origin_ids)].reset_index(drop=True)

    max_fake = int(len(fold_train_real) * cfg.MAX_FAKE_RATIO)
    if max_fake > 0 and len(fold_aug) > max_fake:
        fold_aug = fold_aug.sample(n=max_fake, random_state=cfg.SEED).reset_index(drop=True)

    return pd.concat([fold_train_real, fold_aug], ignore_index=True)


def make_cache_dir(concentration: str, exp_name: str) -> Path:
    band_tag = f"d{cfg.SPECTRAL_DEPTH}"
    size_tag = f"h{cfg.TARGET_HW[0]}_w{cfg.TARGET_HW[1]}"
    norm_tag = "scalar" if np.asarray(cfg.GLOBAL_MIN).ndim == 0 else "bandwise"
    model_tag = cfg.MODEL_NAME
    cache_dir = cfg.CACHE_ROOT / concentration / model_tag / band_tag / size_tag / norm_tag / exp_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_dataset(records_df: pd.DataFrame, cache_dir: Path):
    return HSI3DDataset(
        records_df,
        selected_bands=cfg.SELECTED_BANDS,
        norm_min=cfg.NORM_MIN,
        norm_max=cfg.NORM_MAX,
        target_hw=cfg.TARGET_HW,
        cache_dir=cache_dir,
        use_cache=True,
    )


def prebuild_cache(records_df: pd.DataFrame, cache_dir: Path, desc: str):
    if not cfg.PREBUILD_CACHE:
        return

    ds = build_dataset(records_df, cache_dir)
    paths = records_df["path"].astype(str).tolist()
    total = len(paths)
    print(f"[INFO] Prebuild 3D cache for {desc}: {total} samples")
    for i, p in enumerate(paths, 1):
        ds.get_or_build_cache(p)
        if i == total or i % 25 == 0:
            print(f"[INFO] cache {desc}: {i}/{total}")
    print(f"[INFO] Cache ready: {cache_dir}")


def choose_final_epochs(best_epochs):
    if len(best_epochs) == 0:
        return cfg.EPOCHS
    if getattr(cfg, "FINAL_EPOCH_STRATEGY", "mean_best_epoch") == "mean_best_epoch":
        return max(int(round(float(np.mean(best_epochs)))), 1)
    return cfg.EPOCHS


def train_final_and_test(exp_name, train_real_df, test_real_df, aug_df, use_aug,
                         result_root, chosen_epochs, concentration, batch_size):
    exp_dir = result_root / exp_name
    amp_enabled = (cfg.DEVICE == "cuda") and cfg.USE_AMP

    final_train_df = make_fold_train_df(train_real_df, aug_df, use_aug)
    final_test_df = test_real_df.copy().reset_index(drop=True)

    print(f"\n[Final Train -> Test 3D] concentration={concentration} exp={exp_name} batch_size={batch_size}")
    print("final train real:", len(train_real_df))
    print("final train used:", len(final_train_df))
    print("final test:", len(final_test_df))
    print("chosen epochs:", chosen_epochs)

    cache_train = make_cache_dir(concentration, f"{exp_name}_final_train")
    cache_test = make_cache_dir(concentration, f"{exp_name}_final_test")

    prebuild_cache(final_train_df, cache_train, desc=f"{concentration} {exp_name} final train")
    prebuild_cache(final_test_df, cache_test, desc=f"{concentration} {exp_name} final test")

    ds_train = build_dataset(final_train_df, cache_train)
    ds_test = build_dataset(final_test_df, cache_test)

    dl_train = make_loader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = make_loader(ds_test, batch_size=batch_size, shuffle=False)

    model = make_model()
    print("[INFO] final 3D model device:", next(model.parameters()).device)
    optimizer = make_optimizer(model)
    criterion = make_criterion()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    final_epoch_rows = []
    for epoch in range(1, chosen_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, dl_train, optimizer, criterion, cfg.DEVICE,
            scaler=scaler, amp_enabled=amp_enabled,
            debug_first_batch=(epoch == 1),
        )
        final_epoch_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
        })
        print(f"[Final-3D] Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f}")

    torch.save(model.state_dict(), exp_dir / "final_model_for_test.pth")
    pd.DataFrame(final_epoch_rows).to_csv(exp_dir / "final_train_epochs.csv", index=False)

    test_loss, test_acc = evaluate(
        model, dl_test, criterion, cfg.DEVICE,
        amp_enabled=amp_enabled,
        debug_first_batch=True,
    )
    pred_pack = collect_predictions(model, dl_test, cfg.DEVICE, amp_enabled=amp_enabled)
    test_metrics = compute_binary_metrics(pred_pack["y_true"], pred_pack["y_pred"])
    test_metrics["test_loss"] = float(test_loss)
    test_metrics["acc_eval"] = float(test_acc)

    pred_df = final_test_df.copy().reset_index(drop=True)
    pred_df["y_true"] = pred_pack["y_true"]
    pred_df["y_pred"] = pred_pack["y_pred"]
    pred_df["prob_pos"] = pred_pack["prob_pos"]
    pred_df.to_csv(exp_dir / "test_predictions.csv", index=False)

    test_summary_lines = [
        f"experiment={exp_name}",
        f"concentration={concentration}",
        f"batch_size={batch_size}",
        f"chosen_epochs={chosen_epochs}",
        f"test_loss={test_metrics['test_loss']:.6f}",
        f"ACC(Test)={test_metrics['acc']:.6f}",
        f"SP={test_metrics['sp']:.6f}",
        f"SN={test_metrics['sn']:.6f}",
        f"MCC={test_metrics['mcc']:.6f}",
        f"TP={test_metrics['tp']}",
        f"TN={test_metrics['tn']}",
        f"FP={test_metrics['fp']}",
        f"FN={test_metrics['fn']}",
    ]
    save_text(exp_dir / "test_summary.txt", "\n".join(test_summary_lines))

    return {
        "train_used": len(final_train_df),
        "chosen_epochs": int(chosen_epochs),
        "test_metrics": test_metrics,
    }


def run_experiment(use_aug: bool, train_real_df: pd.DataFrame, test_real_df: pd.DataFrame,
                   aug_df: pd.DataFrame, result_root: Path, concentration: str, batch_size: int):
    exp_name = "aug" if use_aug else "no_aug"
    print("=" * 100)
    print(f"3D Experiment: concentration={concentration} | exp={exp_name} | batch_size={batch_size}")
    print("=" * 100)

    exp_dir = result_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    X = np.arange(len(train_real_df))
    y = train_real_df["label"].values
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)

    fold_best_accs = []
    fold_best_epochs = []
    fold_rows = []
    amp_enabled = (cfg.DEVICE == "cuda") and cfg.USE_AMP
    oof_pred_df_list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{cfg.N_FOLDS}] concentration={concentration} exp={exp_name} batch_size={batch_size}")

        fold_train_real = train_real_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_real = train_real_df.iloc[va_idx].reset_index(drop=True)
        fold_train = make_fold_train_df(fold_train_real, aug_df, use_aug)

        print("train real:", len(fold_train_real))
        print("train final:", len(fold_train))
        print("val real:", len(fold_val_real))
        print("train label counts:\n", fold_train["label"].value_counts(dropna=False))
        print("val label counts:\n", fold_val_real["label"].value_counts(dropna=False))

        fold_cache_train = make_cache_dir(concentration, f"{exp_name}_fold{fold}_train")
        fold_cache_val = make_cache_dir(concentration, f"{exp_name}_fold{fold}_val")

        prebuild_cache(fold_train, fold_cache_train, desc=f"{concentration} {exp_name} fold{fold} train")
        prebuild_cache(fold_val_real, fold_cache_val, desc=f"{concentration} {exp_name} fold{fold} val")

        ds_train = build_dataset(fold_train, fold_cache_train)
        ds_val = build_dataset(fold_val_real, fold_cache_val)

        dl_train = make_loader(ds_train, batch_size=batch_size, shuffle=True)
        dl_val = make_loader(ds_val, batch_size=batch_size, shuffle=False)

        model = make_model()
        print("[INFO] 3D model device:", next(model.parameters()).device)

        optimizer = make_optimizer(model)
        criterion = make_criterion()
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val_acc = -1.0
        best_epoch = -1
        epoch_rows = []
        best_path = exp_dir / f"best_fold{fold}.pth"

        for epoch in range(1, cfg.EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, dl_train, optimizer, criterion, cfg.DEVICE,
                scaler=scaler, amp_enabled=amp_enabled,
                debug_first_batch=(epoch == 1),
            )
            va_loss, va_acc = evaluate(
                model, dl_val, criterion, cfg.DEVICE,
                amp_enabled=amp_enabled,
                debug_first_batch=(epoch == 1),
            )

            epoch_rows.append({
                "epoch": epoch,
                "batch_size": batch_size,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            })

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)

            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f}"
            )

        pd.DataFrame(epoch_rows).to_csv(exp_dir / f"fold{fold}_epochs.csv", index=False)

        best_model = make_model()
        best_model.load_state_dict(torch.load(best_path, map_location=cfg.DEVICE))
        pred_pack = collect_predictions(best_model, dl_val, cfg.DEVICE, amp_enabled=amp_enabled)
        val_metrics = compute_binary_metrics(pred_pack["y_true"], pred_pack["y_pred"])

        oof_fold_df = fold_val_real.copy().reset_index(drop=True)
        oof_fold_df["fold"] = fold
        oof_fold_df["y_true"] = pred_pack["y_true"]
        oof_fold_df["y_pred"] = pred_pack["y_pred"]
        oof_fold_df["prob_pos"] = pred_pack["prob_pos"]
        oof_pred_df_list.append(oof_fold_df)
        oof_fold_df.to_csv(exp_dir / f"fold{fold}_val_predictions.csv", index=False)

        fold_best_accs.append(float(best_val_acc))
        fold_best_epochs.append(int(best_epoch))
        fold_rows.append({
            "fold": fold,
            "concentration": concentration,
            "batch_size": batch_size,
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "val_acc_from_best_model": val_metrics["acc"],
            "val_sp_from_best_model": val_metrics["sp"],
            "val_sn_from_best_model": val_metrics["sn"],
            "val_mcc_from_best_model": val_metrics["mcc"],
            "tp": val_metrics["tp"],
            "tn": val_metrics["tn"],
            "fp": val_metrics["fp"],
            "fn": val_metrics["fn"],
            "n_train_real": len(fold_train_real),
            "n_train_final": len(fold_train),
            "n_val_real": len(fold_val_real),
            "cache_train_dir": str(fold_cache_train),
            "cache_val_dir": str(fold_cache_val),
        })

        del model, best_model, optimizer, scaler, ds_train, ds_val, dl_train, dl_val
        release_memory()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(exp_dir / "fold_summary.csv", index=False)

    oof_df = pd.concat(oof_pred_df_list, ignore_index=True) if len(oof_pred_df_list) > 0 else pd.DataFrame()
    if len(oof_df) > 0:
        oof_metrics = compute_binary_metrics(oof_df["y_true"].values, oof_df["y_pred"].values)
        oof_df.to_csv(exp_dir / "cv5_oof_predictions.csv", index=False)
    else:
        oof_metrics = {"acc": float("nan"), "sp": float("nan"), "sn": float("nan"), "mcc": float("nan"),
                       "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    mean_acc = float(np.mean(fold_best_accs)) if len(fold_best_accs) > 0 else float("nan")
    std_acc = float(np.std(fold_best_accs)) if len(fold_best_accs) > 0 else float("nan")
    chosen_epochs = choose_final_epochs(fold_best_epochs)

    summary_txt = (
        f"experiment = {exp_name}\n"
        f"concentration = {concentration}\n"
        f"batch_size = {batch_size}\n"
        f"mean_best_val_acc = {mean_acc:.6f}\n"
        f"std_best_val_acc = {std_acc:.6f}\n"
        f"fold_best_val_accs = {fold_best_accs}\n"
        f"fold_best_epochs = {fold_best_epochs}\n"
        f"ACC(CV5) = {oof_metrics['acc']:.6f}\n"
        f"CV5_SP = {oof_metrics['sp']:.6f}\n"
        f"CV5_SN = {oof_metrics['sn']:.6f}\n"
        f"CV5_MCC = {oof_metrics['mcc']:.6f}\n"
        f"chosen_final_epochs = {chosen_epochs}\n"
    )
    save_text(exp_dir / "summary.txt", summary_txt)

    print("\n[3D CV5 Summary]")
    print("mean best val acc:", mean_acc)
    print("std best val acc :", std_acc)
    print("ACC(CV5)         :", oof_metrics["acc"])
    print("CV5_SP           :", oof_metrics["sp"])
    print("CV5_SN           :", oof_metrics["sn"])
    print("CV5_MCC          :", oof_metrics["mcc"])
    print("chosen final epochs:", chosen_epochs)

    final_out = train_final_and_test(
        exp_name=exp_name,
        train_real_df=train_real_df,
        test_real_df=test_real_df,
        aug_df=aug_df,
        use_aug=use_aug,
        result_root=result_root,
        chosen_epochs=chosen_epochs,
        concentration=concentration,
        batch_size=batch_size,
    )

    print("\n[3D Test Summary]")
    print("ACC(Test):", final_out["test_metrics"]["acc"])
    print("SP       :", final_out["test_metrics"]["sp"])
    print("SN       :", final_out["test_metrics"]["sn"])
    print("MCC      :", final_out["test_metrics"]["mcc"])

    return {
        "exp_name": exp_name,
        "fold_accs": fold_best_accs,
        "fold_best_epochs": fold_best_epochs,
        "mean_best_val_acc": mean_acc,
        "std_best_val_acc": std_acc,
        "cv5_metrics": oof_metrics,
        "chosen_epochs": chosen_epochs,
        "test_metrics": final_out["test_metrics"],
        "n_final_train": final_out["train_used"],
    }


def write_compare_summary(result_root: Path, concentration: str, batch_size: int, results: dict):
    final_rows = []
    for exp_name, out in results.items():
        final_rows.append({
            "concentration": concentration,
            "batch_size": batch_size,
            "experiment": exp_name,
            "ACC(Test)": out["test_metrics"]["acc"],
            "ACC(CV5)": out["cv5_metrics"]["acc"],
            "SP": out["test_metrics"]["sp"],
            "SN": out["test_metrics"]["sn"],
            "MCC": out["test_metrics"]["mcc"],
            "mean_best_val_acc": out["mean_best_val_acc"],
            "std_best_val_acc": out["std_best_val_acc"],
            "chosen_final_epochs": out["chosen_epochs"],
            "n_final_train": out["n_final_train"],
            "CV5_SP": out["cv5_metrics"]["sp"],
            "CV5_SN": out["cv5_metrics"]["sn"],
            "CV5_MCC": out["cv5_metrics"]["mcc"],
        })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(result_root / "compare_summary.csv", index=False)

    compact_df = final_df[["concentration", "batch_size", "experiment", "ACC(Test)", "ACC(CV5)", "SP", "SN", "MCC"]].copy()
    compact_df.to_csv(result_root / "compare_summary_compact.csv", index=False)

    print("\n[INFO] 3D compare_summary saved to:", result_root / "compare_summary.csv")
    print(final_df)
    return final_df


def run_one_batch_size(train_real_df: pd.DataFrame, test_real_df: pd.DataFrame, aug_df: pd.DataFrame,
                       conc_root: Path, concentration: str, batch_size: int):
    print("\n" + "#" * 100)
    print(f"[3D Grid] concentration={concentration} | batch_size={batch_size}")
    print("#" * 100)

    bs_root = conc_root / f"bs_{batch_size}"
    bs_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for use_aug in cfg.RUN_COMPARE_AUG:
        out = run_experiment(
            use_aug=use_aug,
            train_real_df=train_real_df,
            test_real_df=test_real_df,
            aug_df=aug_df,
            result_root=bs_root,
            concentration=concentration,
            batch_size=batch_size,
        )
        results[out["exp_name"]] = out

    return write_compare_summary(bs_root, concentration, batch_size, results)


def load_aug_records(env):
    if env["aug_csv"].exists():
        print("[INFO] Load existing aug manifest:", env["aug_csv"])
        return pd.read_csv(env["aug_csv"])

    print("[INFO] aug_manifest.csv 不存在，改為掃描 augmentation 資料夾建立")
    aug_df = build_aug_records_from_folders(
        env["aug_root"],
        trainA_fakeB_dir=cfg.TRAIN_AUG_DIR_A2B,
        trainB_fakeA_dir=cfg.TRAIN_AUG_DIR_B2A,
        a_label=cfg.A_LABEL,
        b_label=cfg.B_LABEL,
    )
    env["aug_root"].mkdir(parents=True, exist_ok=True)
    aug_df.to_csv(env["aug_csv"], index=False)
    print("[INFO] Saved aug manifest:", env["aug_csv"])
    return aug_df


def save_env_snapshot(path: Path, env, batch_size=None):
    lines = [
        f"concentration={env['concentration']}",
        f"DEVICE={cfg.DEVICE}",
        f"torch.cuda.is_available()={torch.cuda.is_available()}",
        f"torch.cuda.device_count()={torch.cuda.device_count()}",
        f"MODEL_NAME={cfg.MODEL_NAME}",
        f"TRAIN_CSV={env['train_csv']}",
        f"TEST_CSV={env['test_csv']}",
        f"BASE_DATA={env['base_data']}",
        f"AUG_ROOT={env['aug_root']}",
        f"AUG_CSV={env['aug_csv']}",
        f"CACHE_ROOT={cfg.CACHE_ROOT}",
        f"PREBUILD_CACHE={cfg.PREBUILD_CACHE}",
        f"TARGET_HW={cfg.TARGET_HW}",
        f"SELECTED_BANDS={cfg.SELECTED_BANDS}",
        f"SPECTRAL_DEPTH={cfg.SPECTRAL_DEPTH}",
        f"INPUT_CHANNELS_3D={cfg.INPUT_CHANNELS_3D}",
        f"NUM_CLASSES={cfg.NUM_CLASSES}",
        f"BATCH_SIZE={batch_size if batch_size is not None else cfg.BATCH_SIZE}",
        f"BATCH_SIZES={cfg.BATCH_SIZES}",
        f"LR={cfg.LR}",
        f"EPOCHS={cfg.EPOCHS}",
        f"N_FOLDS={cfg.N_FOLDS}",
        f"SEED={cfg.SEED}",
        f"PRETRAINED={cfg.PRETRAINED}",
        f"USE_AMP={cfg.USE_AMP}",
        f"MAX_FAKE_RATIO={cfg.MAX_FAKE_RATIO}",
        f"FINAL_EPOCH_STRATEGY={cfg.FINAL_EPOCH_STRATEGY}",
    ]
    save_text(path, "\n".join(lines))


def main():
    seed_everything(cfg.SEED)

    if cfg.DEVICE == "cuda" and cfg.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    print_runtime_info()
    master_root = build_master_result_root()

    master_rows = []

    for concentration in cfg.CONCENTRATIONS:
        env = resolve_concentration_env(concentration)

        print("\n" + "=" * 100)
        print(f"[3D Concentration] {concentration}")
        print("=" * 100)

        conc_root = master_root / concentration
        conc_root.mkdir(parents=True, exist_ok=True)

        train_real_df = build_real_records(env["train_csv"], env["base_data"])
        test_real_df = build_real_records(env["test_csv"], env["base_data"])
        aug_df = load_aug_records(env)

        train_real_df.to_csv(conc_root / "train_real_records.csv", index=False)
        test_real_df.to_csv(conc_root / "test_real_records.csv", index=False)
        aug_df.to_csv(conc_root / "aug_records.csv", index=False)

        save_env_snapshot(conc_root / "config_snapshot.txt", env)

        for batch_size in cfg.BATCH_SIZES:
            save_env_snapshot(conc_root / f"config_snapshot_bs_{batch_size}.txt", env, batch_size=batch_size)
            summary_df = run_one_batch_size(
                train_real_df=train_real_df,
                test_real_df=test_real_df,
                aug_df=aug_df,
                conc_root=conc_root,
                concentration=concentration,
                batch_size=batch_size,
            )
            master_rows.append(summary_df)

    if len(master_rows) > 0:
        master_df = pd.concat(master_rows, ignore_index=True)
    else:
        master_df = pd.DataFrame(columns=[
            "concentration", "batch_size", "experiment",
            "ACC(Test)", "ACC(CV5)", "SP", "SN", "MCC"
        ])

    master_df.to_csv(master_root / "grid_compare_summary.csv", index=False)

    compact_cols = ["concentration", "batch_size", "experiment", "ACC(Test)", "ACC(CV5)", "SP", "SN", "MCC"]
    compact_df = master_df[compact_cols].copy()
    compact_df.to_csv(master_root / "grid_compare_summary_compact.csv", index=False)

    print("\n" + "=" * 100)
    print("[DONE-3D] grid_compare_summary.csv")
    print(master_root / "grid_compare_summary.csv")
    print(master_df)
    print("\n[DONE-3D] grid_compare_summary_compact.csv")
    print(master_root / "grid_compare_summary_compact.csv")
    print(compact_df)


if __name__ == "__main__":
    main()
