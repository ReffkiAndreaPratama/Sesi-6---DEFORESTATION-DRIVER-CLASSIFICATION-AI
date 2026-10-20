"""
ForestWatch AI — scripts/predict.py
Menjalankan inferensi pada file raster atau patch tunggal.

Jalankan:
    python scripts/predict.py --input data/sentinel1/scene.tif
    python scripts/predict.py --input data/s1.tif --mode s1
    python scripts/predict.py --demo
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from config.settings import (
    MODELS_DIR, OUTPUTS_DIR,
    NUM_CHANNELS_S1, NUM_CHANNELS_FUSED,
    CLASS_NAMES,
    PATCH_SIZE, PATCH_OVERLAP, MIN_VALID_RATIO,
    MODEL_FILENAME_S1, MODEL_FILENAME_S2, MODEL_FILENAME_FUSED,
    DEVICE, DEMO_IMAGE_SIZE,
)
from src.model import get_device
from src.inference import (
    DeforestationPredictor,
    save_prediction_csv,
    save_prediction_raster,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PREP_CONFIG = OUTPUTS_DIR / "prep_config.json"
PRED_DIR    = OUTPUTS_DIR / "predictions"


def run_demo(mode: str, device):
    """Prediksi pada raster sintetis (mode demo)."""
    from src.data_loader import generate_synthetic_raster
    import tempfile, os

    logger.info("MODE DEMO — prediksi pada raster sintetis")
    H, W = DEMO_IMAGE_SIZE
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_synthetic_raster(
            path       = tmp_path,
            num_bands  = NUM_CHANNELS_FUSED,
            height     = H,
            width      = W,
            seed       = 42,
        )
        run_raster(tmp_path, mode, device, output_suffix="_demo")
    finally:
        os.unlink(tmp_path)


def run_raster(input_path: str, mode: str, device, output_suffix: str = ""):
    fn_map = {"s1": MODEL_FILENAME_S1, "s2": MODEL_FILENAME_S2, "fused": MODEL_FILENAME_FUSED}
    model_path = MODELS_DIR / fn_map[mode]

    if not model_path.exists():
        logger.error(
            "Model '%s' tidak ditemukan: %s\n"
            "Latih model terlebih dahulu: python scripts/train_model.py --mode %s",
            mode, model_path, mode
        )
        sys.exit(1)

    logger.info("Memuat model: %s", model_path.name)
    predictor = DeforestationPredictor.from_checkpoint(
        model_path       = str(model_path),
        prep_config_path = str(PREP_CONFIG) if PREP_CONFIG.exists() else None,
        class_names      = CLASS_NAMES,
        device           = str(device),
        mode             = mode,
        num_channels_s1  = NUM_CHANNELS_S1,
    )

    logger.info("Memuat raster: %s", input_path)
    try:
        from src.data_loader import read_raster
        data, meta = read_raster(input_path, dtype="float32")
        transform  = meta.get("transform")
        crs        = meta.get("crs") or "EPSG:4326"
    except Exception as exc:
        logger.error("Gagal membaca raster: %s", exc)
        sys.exit(1)

    logger.info("Menjalankan inferensi ...")
    result = predictor.predict_raster(
        data            = data,
        patch_size      = PATCH_SIZE,
        overlap         = PATCH_OVERLAP,
        nodata_value    = 0.0,
        min_valid_ratio = MIN_VALID_RATIO,
        transform       = transform,
    )

    logger.info("Patch diproses: %d", result["num_patches"])

    # Ringkasan prediksi
    preds     = [r["class_idx"] for r in result["patch_results"] if r.get("class_idx", -1) >= 0]
    if preds:
        unique, counts = np.unique(preds, return_counts=True)
        logger.info("Distribusi prediksi:")
        for u, c in zip(unique, counts):
            name = CLASS_NAMES[u] if u < len(CLASS_NAMES) else str(u)
            pct  = 100 * c / len(preds)
            logger.info("  %-35s: %4d (%.1f%%)", name, c, pct)

    # Simpan output
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    csv_path = PRED_DIR / f"{stem}{output_suffix}_predictions.csv"
    save_prediction_csv(result["patch_results"], csv_path)
    logger.info("CSV prediksi: %s", csv_path)

    if transform is not None:
        tif_path = PRED_DIR / f"{stem}{output_suffix}_prediction_map.tif"
        try:
            save_prediction_raster(
                result["prediction_map"], tif_path, transform, crs=crs
            )
            logger.info("GeoTIFF prediksi: %s", tif_path)
        except Exception as exc:
            logger.warning("Tidak bisa menyimpan GeoTIFF: %s", exc)

    logger.info("Prediksi selesai.")


def main():
    parser = argparse.ArgumentParser(description="ForestWatch AI — Inferensi")
    parser.add_argument("--input",  type=str, default=None,
                        help="Path ke file raster input")
    parser.add_argument("--mode",   choices=["s1", "s2", "fused"], default="fused")
    parser.add_argument("--demo",   action="store_true",
                        help="Jalankan pada raster sintetis (demo)")
    args = parser.parse_args()

    device = get_device(DEVICE)

    if args.demo or args.input is None:
        run_demo(args.mode, device)
    else:
        if not Path(args.input).exists():
            logger.error("File tidak ditemukan: %s", args.input)
            sys.exit(1)
        run_raster(args.input, args.mode, device)


if __name__ == "__main__":
    main()
