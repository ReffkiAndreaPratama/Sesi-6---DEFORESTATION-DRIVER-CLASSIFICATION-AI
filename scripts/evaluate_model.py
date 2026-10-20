"""
ForestWatch AI — scripts/evaluate_model.py
Mengevaluasi model yang sudah dilatih pada data pengujian.

Jalankan:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --mode s1
    python scripts/evaluate_model.py --all    # evaluasi semua 3 model sekaligus
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config.settings import (
    MODELS_DIR, OUTPUTS_DIR,
    NUM_CHANNELS_S1, NUM_CHANNELS_S2, NUM_CHANNELS_FUSED,
    NUM_CLASSES, CLASS_NAMES, CLASS_COLORS,
    BATCH_SIZE, RANDOM_SEED,
    MODEL_FILENAME_S1, MODEL_FILENAME_S2, MODEL_FILENAME_FUSED,
    DEVICE,
)
from src.model import load_model, get_device
from src.evaluation import (
    evaluate_model, compute_metrics, build_comparison_table,
    error_analysis, save_evaluation_results,
)
from src.dataset import DeforestationDataset, build_dataloaders
from src.patch_generator import generate_synthetic_patches, split_patches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MANIFEST_PATH   = OUTPUTS_DIR / "manifest.csv"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"


def load_test_patches(mode: str):
    """Memuat patch pengujian dari manifest atau buat sintetis."""
    import pandas as pd
    from src.patch_generator import Patch

    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        df = df[df["split"] == "test"].dropna(subset=["class_idx", "npy_path"])
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
                label_name  = row.get("class_name"),
            ))
        if patches:
            logger.info("Patch test dari manifest: %d", len(patches))
            return patches

    # Fallback sintetis
    logger.warning("Manifest tidak ada — menggunakan patch sintetis untuk evaluasi demo.")
    all_p = generate_synthetic_patches(
        num_patches=120, num_classes=NUM_CLASSES,
        channels=NUM_CHANNELS_FUSED, patch_size=64, seed=RANDOM_SEED + 1
    )
    _, _, test_p = split_patches(all_p, 0.15, 0.15, RANDOM_SEED)
    return test_p


def evaluate_one(mode: str, device: torch.device) -> dict:
    fn_map = {
        "s1"   : MODEL_FILENAME_S1,
        "s2"   : MODEL_FILENAME_S2,
        "fused": MODEL_FILENAME_FUSED,
    }
    model_path = MODELS_DIR / fn_map[mode]

    if not model_path.exists():
        logger.warning("Model tidak ditemukan: %s (skip)", model_path)
        return {"error": f"Model {mode} tidak ditemukan."}

    logger.info("Evaluasi model: %s", model_path.name)
    model, meta = load_model(str(model_path), device=str(device))

    patches = load_test_patches(mode)
    if not patches:
        return {"error": "Tidak ada patch test."}

    test_ds = DeforestationDataset(
        patches, mode=mode,
        num_channels_s1=NUM_CHANNELS_S1, augment=False
    )
    loaders = build_dataloaders(
        test_ds, test_ds,
        batch_size=BATCH_SIZE, balance_classes=False
    )

    metrics = evaluate_model(
        model, loaders["train"], device,
        class_names=CLASS_NAMES,
        return_predictions=True,
    )

    # Error analysis
    if "y_true" in metrics and "y_pred" in metrics:
        metrics["error_analysis"] = error_analysis(
            metrics["y_true"], metrics["y_pred"], CLASS_NAMES
        )

    # Simpan
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(
        metrics,
        EVAL_OUTPUT_DIR / f"eval_{mode}.json",
        experiment_name=f"ForestWatch-{mode.upper()}",
    )

    logger.info(
        "  Accuracy=%.4f | F1=%.4f | Inferensi=%.2f ms",
        metrics.get("accuracy", 0),
        metrics.get("f1_macro", 0),
        metrics.get("avg_inference_ms", 0),
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ForestWatch AI — Evaluasi Model")
    parser.add_argument("--mode", choices=["s1", "s2", "fused"], default="fused")
    parser.add_argument("--all",  action="store_true",
                        help="Evaluasi semua mode (s1, s2, fused)")
    args = parser.parse_args()

    device = get_device(DEVICE)
    modes  = ["s1", "s2", "fused"] if args.all else [args.mode]

    all_results = {}
    for mode in modes:
        logger.info("-" * 50)
        all_results[f"ForestWatch-{mode.upper()}"] = evaluate_one(mode, device)

    # Tabel perbandingan
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if len(valid_results) > 1:
        table = build_comparison_table(valid_results)
        logger.info("\n%s", "TABEL PERBANDINGAN MODEL".center(60))
        logger.info("%-30s %8s %8s %8s", "Eksperimen", "Accuracy", "F1 Macro", "Inf(ms)")
        logger.info("-" * 60)
        for row in table:
            logger.info(
                "%-30s %8.4f %8.4f %8s",
                row.get("Eksperimen", ""),
                row.get("Accuracy", 0),
                row.get("F1 (Macro)", 0),
                row.get("Inferensi (ms)", "—"),
            )

        # Simpan tabel ke JSON
        with open(EVAL_OUTPUT_DIR / "comparison_table.json", "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Tabel perbandingan: %s", EVAL_OUTPUT_DIR / "comparison_table.json")

    logger.info("Evaluasi selesai. Hasil di: %s", EVAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
