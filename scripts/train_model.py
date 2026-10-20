"""
ForestWatch AI — scripts/train_model.py
Melatih model CNN klasifikasi penyebab deforestasi.

Jalankan:
    python scripts/train_model.py
    python scripts/train_model.py --mode s1
    python scripts/train_model.py --mode s2
    python scripts/train_model.py --mode fused --epochs 30 --arch cnn
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from config.settings import (
    OUTPUTS_DIR, MODELS_DIR,
    NUM_CHANNELS_S1, NUM_CHANNELS_S2, NUM_CHANNELS_FUSED,
    NUM_CLASSES, CLASS_NAMES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, EARLY_STOP_PATIENCE, RANDOM_SEED,
    MODEL_FILENAME_S1, MODEL_FILENAME_S2, MODEL_FILENAME_FUSED,
    DEVICE,
)
from src.model import build_model, get_device
from src.train import train_model, compute_class_weights
from src.dataset import DeforestationDataset, build_dataloaders
from src.patch_generator import generate_synthetic_patches, split_patches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MANIFEST_PATH = OUTPUTS_DIR / "manifest.csv"
PLOTS_DIR     = OUTPUTS_DIR / "plots"


def load_patches_from_manifest(manifest_path: Path, mode: str, num_channels_s1: int):
    """Memuat patch dari manifest CSV."""
    import numpy as np
    import pandas as pd
    from src.patch_generator import Patch

    if not manifest_path.exists():
        return None

    df = pd.read_csv(manifest_path)
    df = df.dropna(subset=["class_idx", "npy_path"])
    patches = []
    for _, row in df.iterrows():
        npy_path = Path(row["npy_path"])
        if not npy_path.exists():
            continue
        data = np.load(npy_path).astype(np.float32)
        patches.append(Patch(
            patch_id    = row.get("patch_id", f"p_{len(patches)}"),
            data        = data,
            row_start   = int(row.get("row_start", 0)),
            col_start   = int(row.get("col_start", 0)),
            patch_size  = int(row.get("patch_size", data.shape[-1])),
            valid_ratio = float(row.get("valid_ratio", 1.0)),
            label       = int(row["class_idx"]),
            label_name  = row.get("class_name", None),
            source_path = row.get("source_path", None),
        ))
    return patches


def main():
    parser = argparse.ArgumentParser(description="ForestWatch AI — Pelatihan Model")
    parser.add_argument("--mode",    choices=["s1", "s2", "fused"], default="fused")
    parser.add_argument("--arch",    choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--epochs",  type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--demo",    action="store_true",
                        help="Paksa mode demo (data sintetis)")
    parser.add_argument("--num-patches", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = get_device(DEVICE)

    # ── Tentukan jumlah kanal ────────────────────────────────
    ch_map = {"s1": NUM_CHANNELS_S1, "s2": NUM_CHANNELS_S2, "fused": NUM_CHANNELS_FUSED}
    in_channels = ch_map[args.mode]

    # ── Nama file model output ───────────────────────────────
    fn_map = {"s1": MODEL_FILENAME_S1, "s2": MODEL_FILENAME_S2, "fused": MODEL_FILENAME_FUSED}
    model_path = MODELS_DIR / fn_map[args.mode]

    logger.info("=" * 60)
    logger.info("Pelatihan ForestWatch AI")
    logger.info("  Mode      : %s (%d kanal)", args.mode, in_channels)
    logger.info("  Arsitektur: %s", args.arch)
    logger.info("  Epochs    : %d", args.epochs)
    logger.info("  Device    : %s", device)
    logger.info("  Output    : %s", model_path)
    logger.info("=" * 60)

    # ── Muat atau buat data ──────────────────────────────────
    patches = None
    if not args.demo:
        patches = load_patches_from_manifest(MANIFEST_PATH, args.mode, NUM_CHANNELS_S1)

    if not patches:
        logger.warning(
            "Manifest tidak ditemukan atau kosong. Menggunakan data sintetis.\n"
            "Jalankan 'python scripts/prepare_data.py' terlebih dahulu untuk data nyata."
        )
        patches = generate_synthetic_patches(
            num_patches = args.num_patches,
            num_classes = NUM_CLASSES,
            channels    = NUM_CHANNELS_FUSED,
            patch_size  = 64,
            seed        = RANDOM_SEED,
        )

    train_p, val_p, _ = split_patches(
        patches, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED
    )
    logger.info("Data: train=%d | val=%d", len(train_p), len(val_p))

    # ── Dataset & DataLoader ──────────────────────────────────
    train_ds = DeforestationDataset(train_p, mode=args.mode,
                                    num_channels_s1=NUM_CHANNELS_S1, augment=True)
    val_ds   = DeforestationDataset(val_p,   mode=args.mode,
                                    num_channels_s1=NUM_CHANNELS_S1, augment=False)

    in_channels = train_ds.get_num_channels()
    loaders = build_dataloaders(
        train_ds, val_ds,
        batch_size     = args.batch,
        balance_classes= True,
        seed           = RANDOM_SEED,
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(
        architecture = args.arch,
        in_channels  = in_channels,
        num_classes  = NUM_CLASSES,
    )

    # ── Class weights ─────────────────────────────────────────
    dist    = train_ds.get_class_distribution()
    weights = compute_class_weights(dist, NUM_CLASSES)
    logger.info("Distribusi kelas train: %s", dist)

    # ── Pelatihan ─────────────────────────────────────────────
    history_csv = PLOTS_DIR / f"history_{args.mode}.csv"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, history = train_model(
        model            = model,
        train_loader     = loaders["train"],
        val_loader       = loaders["val"],
        num_epochs       = args.epochs,
        learning_rate    = args.lr,
        weight_decay     = WEIGHT_DECAY,
        lr_step_size     = LR_STEP_SIZE,
        lr_gamma         = LR_GAMMA,
        patience         = EARLY_STOP_PATIENCE,
        device           = device,
        save_path        = model_path,
        history_csv_path = history_csv,
        class_weights    = weights,
    )

    logger.info("=" * 60)
    logger.info("Pelatihan selesai")
    logger.info("  Epoch terbaik : %d", history.best_epoch + 1)
    logger.info("  Val loss      : %.4f", history.best_val_loss)
    logger.info("  Waktu         : %.1f detik", history.training_time_s)
    logger.info("  Model simpan  : %s", model_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
