"""
ForestWatch AI — inference.py
Inferensi model untuk patch tunggal maupun batch raster penuh.

Pipeline inferensi:
  1. Muat model dari checkpoint
  2. Muat konfigurasi preprocessing (agar skala konsisten)
  3. Preprocessing patch/raster dengan parameter yang sama seperti training
  4. Forward pass → prediksi kelas + probabilitas
  5. Opsional: hasilkan peta klasifikasi geospasial
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Prediktor utama
# ──────────────────────────────────────────────────────────────

class DeforestationPredictor:
    """
    Wrapper inferensi untuk model ForestWatch AI.

    Usage
    -----
    predictor = DeforestationPredictor.from_checkpoint(
        model_path="models/model_fused.pt",
        prep_config_path="outputs/prep_config.json",
        device="cpu",
    )
    result = predictor.predict_patch(patch_array)
    """

    def __init__(
        self,
        model: nn.Module,
        prep_config: Optional[Dict] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        mode: str = "fused",
        num_channels_s1: int = 2,
    ):
        self.model           = model.eval()
        self.prep_config     = prep_config or {}
        self.class_names     = class_names
        self.device          = torch.device(device)
        self.mode            = mode
        self.num_channels_s1 = num_channels_s1
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        prep_config_path: Optional[str | Path] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        mode: str = "fused",
        num_channels_s1: int = 2,
    ) -> "DeforestationPredictor":
        """Factory: memuat model dan konfigurasi preprocessing."""
        from src.model import load_model
        model, meta = load_model(model_path, device=device)

        prep_config: Dict = {}
        if prep_config_path is not None:
            from src.raster_preprocessing import load_preprocessing_config
            try:
                prep_config = load_preprocessing_config(prep_config_path)
            except FileNotFoundError:
                logger.warning(
                    "Konfigurasi preprocessing tidak ditemukan: %s. "
                    "Menggunakan normalisasi default.", prep_config_path
                )

        return cls(
            model            = model,
            prep_config      = prep_config,
            class_names      = class_names,
            device           = device,
            mode             = mode,
            num_channels_s1  = num_channels_s1,
        )

    # ── Channel selection ─────────────────────────────────────

    def _select_channels(self, data: np.ndarray) -> np.ndarray:
        if self.mode == "s1":
            return data[:self.num_channels_s1]
        elif self.mode == "s2":
            return data[self.num_channels_s1:]
        return data  # fused

    # ── Single-patch inference ─────────────────────────────────

    def predict_patch(
        self,
        patch: np.ndarray,
        return_proba: bool = True,
    ) -> Dict:
        """
        Prediksi untuk satu patch.

        Parameters
        ----------
        patch : np.ndarray, shape (C, H, W) atau (H, W) untuk 1 kanal
        return_proba : bool
            Jika True, kembalikan probabilitas semua kelas.

        Returns
        -------
        dict:
            class_idx   : int
            class_name  : str | None
            confidence  : float (probabilitas kelas prediksi)
            probabilities : list[float] | None
            inference_ms  : float
        """
        if patch.ndim == 2:
            patch = patch[np.newaxis, ...]

        patch = self._select_channels(patch)
        tensor = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        inference_ms = (time.perf_counter() - t0) * 1000

        class_idx   = int(np.argmax(probs))
        confidence  = float(probs[class_idx])
        class_name  = (
            self.class_names[class_idx]
            if self.class_names and class_idx < len(self.class_names)
            else None
        )

        return {
            "class_idx"     : class_idx,
            "class_name"    : class_name,
            "confidence"    : confidence,
            "probabilities" : probs.tolist() if return_proba else None,
            "inference_ms"  : round(inference_ms, 2),
        }

    # ── Batch inference ────────────────────────────────────────

    def predict_batch(
        self,
        patches: List[np.ndarray],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Prediksi untuk daftar patch secara batch.

        Parameters
        ----------
        patches : list[np.ndarray], masing-masing shape (C, H, W)
        """
        results = []
        for i in range(0, len(patches), batch_size):
            batch_arrays = patches[i:i + batch_size]
            selected     = [
                self._select_channels(p if p.ndim == 3 else p[np.newaxis, ...])
                for p in batch_arrays
            ]
            # Pastikan dimensi sama
            try:
                tensor = torch.from_numpy(
                    np.stack(selected)
                ).float().to(self.device)
            except Exception as exc:
                logger.error("Gagal membuat batch tensor: %s", exc)
                for _ in batch_arrays:
                    results.append({"class_idx": -1, "error": str(exc)})
                continue

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            batch_ms = (time.perf_counter() - t0) * 1000

            for j, p in enumerate(probs):
                class_idx  = int(np.argmax(p))
                confidence = float(p[class_idx])
                class_name = (
                    self.class_names[class_idx]
                    if self.class_names and class_idx < len(self.class_names)
                    else None
                )
                results.append({
                    "class_idx"     : class_idx,
                    "class_name"    : class_name,
                    "confidence"    : confidence,
                    "probabilities" : p.tolist(),
                    "inference_ms"  : round(batch_ms / len(batch_arrays), 2),
                })

        return results

    # ── Full raster inference ──────────────────────────────────

    def predict_raster(
        self,
        data: np.ndarray,
        patch_size: int = 64,
        overlap: float = 0.25,
        nodata_value: float = 0.0,
        min_valid_ratio: float = 0.8,
        transform=None,
    ) -> Dict:
        """
        Melakukan klasifikasi pada seluruh raster dengan sliding window.

        Returns
        -------
        dict:
            prediction_map  : np.ndarray (H, W) — peta kelas
            confidence_map  : np.ndarray (H, W) — peta kepercayaan
            patch_results   : list[dict]         — hasil tiap patch
            num_patches     : int
            transform       : rasterio Affine | None
        """
        from src.patch_generator import generate_patches

        patches = generate_patches(
            data            = data,
            patch_size      = patch_size,
            overlap         = overlap,
            nodata_value    = nodata_value,
            min_valid_ratio = min_valid_ratio,
            transform       = transform,
        )

        if not patches:
            logger.warning("Tidak ada patch valid yang dihasilkan dari raster.")
            C, H, W = data.shape if data.ndim == 3 else (1,) + data.shape
            return {
                "prediction_map" : np.full((H, W), -1, dtype=np.int16),
                "confidence_map" : np.zeros((H, W), dtype=np.float32),
                "patch_results"  : [],
                "num_patches"    : 0,
                "transform"      : transform,
            }

        # Prediksi semua patch
        patch_arrays = [p.data for p in patches]
        results      = self.predict_batch(patch_arrays)

        # Buat peta prediksi (dimensi raster)
        C, H, W = data.shape if data.ndim == 3 else (1,) + data.shape
        pred_map = np.full((H, W), -1, dtype=np.int16)
        conf_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        for patch, res in zip(patches, results):
            r  = patch.row_start
            c  = patch.col_start
            ps = patch.patch_size
            r2 = min(r + ps, H)
            c2 = min(c + ps, W)

            cls_idx = res.get("class_idx", -1)
            conf    = res.get("confidence", 0.0)

            # Pengisian berdasarkan kepercayaan tertinggi
            mask = conf_map[r:r2, c:c2] < conf
            pred_map[r:r2, c:c2][mask] = cls_idx
            conf_map[r:r2, c:c2][mask] = conf
            count_map[r:r2, c:c2] += 1

        return {
            "prediction_map" : pred_map,
            "confidence_map" : conf_map,
            "patch_results"  : results,
            "num_patches"    : len(patches),
            "transform"      : transform,
        }


# ──────────────────────────────────────────────────────────────
# Ekspor hasil prediksi
# ──────────────────────────────────────────────────────────────

def save_prediction_raster(
    prediction_map: np.ndarray,
    output_path: str | Path,
    transform,
    crs: str = "EPSG:4326",
    nodata: int = -1,
) -> Path:
    """Menyimpan peta prediksi sebagai GeoTIFF."""
    try:
        import rasterio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path, "w",
            driver    = "GTiff",
            height    = prediction_map.shape[0],
            width     = prediction_map.shape[1],
            count     = 1,
            dtype     = "int16",
            crs       = crs,
            transform = transform,
            nodata    = nodata,
        ) as dst:
            dst.write(prediction_map.astype("int16"), 1)

        logger.info("Peta prediksi disimpan: %s", output_path)
        return output_path
    except ImportError:
        raise RuntimeError("rasterio diperlukan untuk menyimpan GeoTIFF.")


def save_prediction_csv(
    patch_results: List[Dict],
    output_path: str | Path,
    patches=None,
) -> Path:
    """Menyimpan hasil prediksi patch ke CSV."""
    import pandas as pd
    rows = []
    for i, res in enumerate(patch_results):
        row = {
            "patch_idx"  : i,
            "class_idx"  : res.get("class_idx", -1),
            "class_name" : res.get("class_name", ""),
            "confidence" : res.get("confidence", 0.0),
        }
        if patches and i < len(patches):
            row["row_start"] = patches[i].row_start
            row["col_start"] = patches[i].col_start
            if patches[i].geo_bounds:
                row["left"]   = patches[i].geo_bounds[0]
                row["bottom"] = patches[i].geo_bounds[1]
                row["right"]  = patches[i].geo_bounds[2]
                row["top"]    = patches[i].geo_bounds[3]
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Prediksi CSV disimpan: %s (%d baris)", output_path, len(df))
    return output_path
