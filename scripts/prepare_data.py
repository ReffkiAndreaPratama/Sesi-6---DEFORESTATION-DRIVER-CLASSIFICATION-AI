"""
ForestWatch AI — scripts/prepare_data.py
Menyiapkan dataset: memindai file, membuat citra sintetis (mode demo),
preprocessing, membuat patch, dan menyimpan manifest CSV.

Jalankan:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --demo          # paksa mode demo
    python scripts/prepare_data.py --s1 data/s1    # tentukan direktori custom
"""

import argparse
import logging
import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import (
    SENTINEL1_DIR, SENTINEL2_DIR, LABELS_DIR,
    OUTPUTS_DIR, MODELS_DIR,
    S1_BANDS, S2_BANDS, S1_NODATA, S2_NODATA,
    S1_VALID_RANGE, S2_VALID_RANGE, S2_SCALE_FACTOR,
    PATCH_SIZE, PATCH_OVERLAP, MIN_VALID_RATIO,
    NUM_CHANNELS_S1, NUM_CHANNELS_S2,
    DEFORESTATION_CLASSES, CLASS_NAMES,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, DEMO_MODE,
    DEMO_IMAGE_SIZE,
)
from src.data_loader import (
    build_dataset_inventory, generate_synthetic_raster,
    read_raster, save_inventory_json,
)
from src.raster_preprocessing import (
    preprocess_sentinel1, preprocess_sentinel2,
    fuse_s1_s2, save_preprocessing_config,
)
from src.patch_generator import (
    generate_patches, generate_synthetic_patches,
    assign_labels_from_raster, split_patches, save_patches,
)
from src.dataset import save_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PATCHES_DIR = OUTPUTS_DIR / "patches"
MANIFEST_PATH = OUTPUTS_DIR / "manifest.csv"
PREP_CONFIG_PATH = OUTPUTS_DIR / "prep_config.json"
INVENTORY_PATH = OUTPUTS_DIR / "inventory.json"


def run_demo_mode(args):
    """Jalankan pipeline dengan data sintetis untuk demonstrasi."""
    logger.info("=" * 60)
    logger.info("MODE DEMO — menggunakan data sintetis")
    logger.info("Data sintetis TIDAK mewakili kondisi deforestasi nyata.")
    logger.info("=" * 60)

    H, W = DEMO_IMAGE_SIZE
    num_patches = args.num_patches
    num_classes = len(DEFORESTATION_CLASSES)

    logger.info("Membuat %d patch sintetis (%d kelas) ...", num_patches, num_classes)
    patches = generate_synthetic_patches(
        num_patches = num_patches,
        num_classes = num_classes,
        channels    = NUM_CHANNELS_S1 + NUM_CHANNELS_S2,
        patch_size  = PATCH_SIZE,
        seed        = RANDOM_SEED,
    )

    # Split
    train_p, val_p, test_p = split_patches(
        patches,
        val_ratio  = VAL_SPLIT,
        test_ratio = TEST_SPLIT,
        seed       = RANDOM_SEED,
        stratify   = True,
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(train_p), len(val_p), len(test_p)
    )

    # Simpan patch
    manifest_rows = []
    for split_name, split_patches_list in [
        ("train", train_p), ("val", val_p), ("test", test_p)
    ]:
        rows = save_patches(split_patches_list, PATCHES_DIR, split=split_name)
        manifest_rows.extend(rows)

    save_manifest(manifest_rows, MANIFEST_PATH)
    logger.info("Manifest disimpan: %s (%d baris)", MANIFEST_PATH, len(manifest_rows))

    # Simpan konfigurasi preprocessing dummy
    prep_config = {
        "mode"     : "demo",
        "s1_stats" : {f"band_{i}": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2}
                      for i in range(NUM_CHANNELS_S1)},
        "s2_stats" : {f"band_{i}": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2}
                      for i in range(NUM_CHANNELS_S2)},
    }
    save_preprocessing_config(prep_config, PREP_CONFIG_PATH)
    logger.info("Persiapan data selesai (mode demo).")
    _print_summary(manifest_rows)


def run_real_mode(args):
    """Jalankan pipeline dengan data nyata."""
    s1_dir = Path(args.s1) if args.s1 else SENTINEL1_DIR
    s2_dir = Path(args.s2) if args.s2 else SENTINEL2_DIR
    lb_dir = Path(args.labels) if args.labels else LABELS_DIR

    logger.info("Memindai direktori dataset ...")
    inventory = build_dataset_inventory(s1_dir, s2_dir, lb_dir)
    save_inventory_json(inventory, INVENTORY_PATH)

    logger.info(
        "Inventori: S1=%d | S2=%d | Label=%d",
        inventory["summary"]["sentinel1_count"],
        inventory["summary"]["sentinel2_count"],
        inventory["summary"]["label_count"],
    )

    if not inventory["has_data"]:
        logger.warning(
            "Dataset tidak ditemukan di direktori yang ditentukan.\n"
            "Gunakan flag --demo untuk mode demonstrasi, atau\n"
            "letakkan file raster di:\n"
            "  %s\n  %s\n  %s",
            s1_dir, s2_dir, lb_dir
        )
        sys.exit(1)

    all_patches = []
    prep_config_combined = {}

    # Proses file S1 + S2 yang berpasangan (heuristik: urutan file)
    s1_files = inventory["s1_files"]
    s2_files = inventory["s2_files"]
    n_pairs  = min(len(s1_files), len(s2_files))

    if n_pairs == 0:
        logger.error("Tidak ada pasangan S1+S2 yang ditemukan.")
        sys.exit(1)

    logger.info("Memproses %d pasangan S1+S2 ...", n_pairs)

    for idx in range(n_pairs):
        s1_path = s1_files[idx]
        s2_path = s2_files[idx]
        logger.info("[%d/%d] S1: %s", idx + 1, n_pairs, Path(s1_path).name)
        logger.info("[%d/%d] S2: %s", idx + 1, n_pairs, Path(s2_path).name)

        try:
            s1_data, s1_meta = read_raster(s1_path, dtype="float32")
            s2_data, s2_meta = read_raster(s2_path, dtype="float32")
        except Exception as exc:
            logger.error("Gagal membaca raster: %s", exc)
            continue

        # Preprocessing
        s1_proc, s1_stats = preprocess_sentinel1(
            s1_data,
            nodata_value = S1_NODATA,
            valid_range  = S1_VALID_RANGE,
        )
        s2_proc, s2_stats = preprocess_sentinel2(
            s2_data,
            nodata_value = S2_NODATA,
            scale_factor = S2_SCALE_FACTOR,
            valid_range  = S2_VALID_RANGE,
        )

        # Gabungkan kanal
        try:
            fused = fuse_s1_s2(s1_proc, s2_proc)
        except ValueError as exc:
            logger.warning("Tidak bisa menggabungkan S1+S2: %s. Melewati.", exc)
            continue

        prep_config_combined[f"pair_{idx}_s1"] = s1_stats
        prep_config_combined[f"pair_{idx}_s2"] = s2_stats

        # Buat patch
        transform = s1_meta.get("transform")
        patches = generate_patches(
            fused,
            patch_size      = PATCH_SIZE,
            overlap         = PATCH_OVERLAP,
            nodata_value    = 0.0,
            min_valid_ratio = MIN_VALID_RATIO,
            source_path     = s1_path,
            transform       = transform,
        )
        logger.info("  Patch dibuat: %d", len(patches))
        all_patches.extend(patches)

    if not all_patches:
        logger.error("Tidak ada patch yang berhasil dibuat.")
        sys.exit(1)

    # Split & simpan
    train_p, val_p, test_p = split_patches(
        all_patches, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(train_p), len(val_p), len(test_p)
    )

    manifest_rows = []
    for split_name, sp in [("train", train_p), ("val", val_p), ("test", test_p)]:
        rows = save_patches(sp, PATCHES_DIR, split=split_name)
        manifest_rows.extend(rows)

    save_manifest(manifest_rows, MANIFEST_PATH)
    save_preprocessing_config(prep_config_combined, PREP_CONFIG_PATH)
    logger.info("Persiapan data selesai.")
    _print_summary(manifest_rows)


def _print_summary(manifest_rows):
    df = pd.DataFrame(manifest_rows)
    logger.info("-" * 40)
    logger.info("RINGKASAN DATASET")
    logger.info("  Total patch  : %d", len(df))
    for split in ["train", "val", "test"]:
        n = len(df[df["split"] == split]) if "split" in df.columns else 0
        logger.info("  %-10s   : %d", split, n)
    if "class_idx" in df.columns:
        labeled = df["class_idx"].notna().sum()
        logger.info("  Berlabel     : %d", int(labeled))
    logger.info("  Manifest     : %s", MANIFEST_PATH)
    logger.info("  Prep config  : %s", PREP_CONFIG_PATH)
    logger.info("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="ForestWatch AI — Persiapan Data")
    parser.add_argument("--demo",    action="store_true",
                        help="Gunakan data sintetis (mode demo)")
    parser.add_argument("--s1",     type=str, default=None,
                        help="Direktori citra Sentinel-1")
    parser.add_argument("--s2",     type=str, default=None,
                        help="Direktori citra Sentinel-2")
    parser.add_argument("--labels", type=str, default=None,
                        help="Direktori file label")
    parser.add_argument("--num-patches", type=int, default=300,
                        help="Jumlah patch sintetis (mode demo, default=300)")
    args = parser.parse_args()

    force_demo = args.demo or DEMO_MODE
    if force_demo:
        run_demo_mode(args)
    else:
        run_real_mode(args)


if __name__ == "__main__":
    main()
