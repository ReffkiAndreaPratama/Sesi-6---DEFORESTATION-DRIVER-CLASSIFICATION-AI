"""
ForestWatch AI — raster_preprocessing.py
Pipeline preprocessing modular untuk citra Sentinel-1 dan Sentinel-2.

Urutan preprocessing:
  1. Kliping nilai ke rentang valid
  2. Penanganan NoData (mask + interpolasi sederhana)
  3. Normalisasi per-kanal
  4. Penyamaan resolusi spasial (opsional)

Semua parameter preprocessing disimpan ke JSON agar inferensi
menggunakan skala yang identik dengan pelatihan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Sentinel-1 preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess_sentinel1(
    data: np.ndarray,
    nodata_value: float = -9999.0,
    valid_range: Tuple[float, float] = (-50.0, 10.0),
    norm_method: str = "minmax",
    stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocessing citra Sentinel-1 (SAR backscatter, nilai dB).

    Parameters
    ----------
    data : np.ndarray, shape (bands, H, W)
    nodata_value : float
        Nilai piksel NoData.
    valid_range : tuple
        Rentang nilai dB yang valid (min, max).
    norm_method : str
        'minmax' → [0, 1] | 'zscore' → mean=0, std=1
    stats : dict | None
        Statistik dari data latih (untuk inferensi). Jika None, dihitung
        dari data input.

    Returns
    -------
    processed : np.ndarray, shape (bands, H, W), dtype float32
    stats     : dict  — statistik yang digunakan (simpan untuk inferensi)
    """
    data = data.astype(np.float32)
    bands, H, W = data.shape

    # Buat mask piksel tidak valid
    nodata_mask = (data == nodata_value) | np.isnan(data) | np.isinf(data)

    # Kliping ke rentang valid (tetap NaN di luar)
    data = np.clip(data, valid_range[0], valid_range[1])
    # Set piksel NoData ke NaN untuk statistik
    data[nodata_mask] = np.nan

    # Hitung atau gunakan statistik yang diberikan
    computed_stats: Dict = {}
    if stats is None:
        stats = {}
        for b in range(bands):
            band_data = data[b]
            valid_vals = band_data[~np.isnan(band_data)]
            if valid_vals.size == 0:
                b_min, b_max, b_mean, b_std = (
                    float(valid_range[0]), float(valid_range[1]), 0.0, 1.0
                )
            else:
                b_min  = float(np.nanmin(valid_vals))
                b_max  = float(np.nanmax(valid_vals))
                b_mean = float(np.nanmean(valid_vals))
                b_std  = float(np.nanstd(valid_vals)) or 1.0
            stats[f"band_{b}"] = {
                "min": b_min, "max": b_max,
                "mean": b_mean, "std": b_std
            }
        computed_stats = stats
    else:
        computed_stats = stats

    # Normalisasi
    processed = np.zeros_like(data)
    for b in range(bands):
        band_data = data[b].copy()
        s = computed_stats[f"band_{b}"]
        if norm_method == "minmax":
            denom = s["max"] - s["min"]
            if denom == 0:
                denom = 1.0
            processed[b] = (band_data - s["min"]) / denom
        elif norm_method == "zscore":
            processed[b] = (band_data - s["mean"]) / s["std"]
        else:
            raise ValueError(f"norm_method tidak dikenal: {norm_method}")

    # Ganti NaN dengan 0 setelah normalisasi
    processed = np.nan_to_num(processed, nan=0.0)
    processed = processed.astype(np.float32)

    result_stats = {
        "sensor"      : "sentinel1",
        "norm_method" : norm_method,
        "valid_range" : list(valid_range),
        "nodata_value": nodata_value,
        "bands"       : computed_stats,
        "nodata_ratio": float(np.mean(nodata_mask)),
    }

    return processed, result_stats


# ──────────────────────────────────────────────────────────────
# Sentinel-2 preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess_sentinel2(
    data: np.ndarray,
    nodata_value: int = 0,
    scale_factor: float = 10000.0,
    valid_range: Tuple[float, float] = (0.0, 1.0),
    norm_method: str = "minmax",
    stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocessing citra Sentinel-2 (reflektansi BOA Level-2A).

    Parameters
    ----------
    scale_factor : float
        Faktor pembagi jika data asli dalam integer (misal 10000).
        Set ke 1.0 jika sudah berupa reflektansi float.
    """
    data = data.astype(np.float32)
    bands, H, W = data.shape

    nodata_mask = (data == nodata_value) | np.isnan(data) | np.isinf(data)

    # Konversi ke reflektansi (0–1)
    data = data / scale_factor
    data = np.clip(data, valid_range[0], valid_range[1])
    data[nodata_mask] = np.nan

    computed_stats: Dict = {}
    if stats is None:
        stats_out: Dict = {}
        for b in range(bands):
            valid_vals = data[b][~np.isnan(data[b])]
            if valid_vals.size == 0:
                b_min, b_max, b_mean, b_std = 0.0, 1.0, 0.5, 0.2
            else:
                b_min  = float(np.nanmin(valid_vals))
                b_max  = float(np.nanmax(valid_vals))
                b_mean = float(np.nanmean(valid_vals))
                b_std  = float(np.nanstd(valid_vals)) or 1e-6
            stats_out[f"band_{b}"] = {
                "min": b_min, "max": b_max,
                "mean": b_mean, "std": b_std
            }
        computed_stats = stats_out
    else:
        computed_stats = stats

    processed = np.zeros_like(data)
    for b in range(bands):
        band_data = data[b].copy()
        s = computed_stats[f"band_{b}"]
        if norm_method == "minmax":
            denom = s["max"] - s["min"]
            if denom == 0:
                denom = 1.0
            processed[b] = (band_data - s["min"]) / denom
        elif norm_method == "zscore":
            processed[b] = (band_data - s["mean"]) / s["std"]

    processed = np.nan_to_num(processed, nan=0.0)
    processed = processed.astype(np.float32)

    result_stats = {
        "sensor"       : "sentinel2",
        "norm_method"  : norm_method,
        "scale_factor" : scale_factor,
        "valid_range"  : list(valid_range),
        "nodata_value" : nodata_value,
        "bands"        : computed_stats,
        "nodata_ratio" : float(np.mean(nodata_mask)),
    }

    return processed, result_stats


# ──────────────────────────────────────────────────────────────
# Penyamaan resolusi spasial (resampling)
# ──────────────────────────────────────────────────────────────

def resample_to_reference(
    src_path: str | Path,
    ref_path: str | Path,
    dst_path: str | Path,
    resampling: str = "bilinear",
) -> Path:
    """
    Meresampling raster sumber (src) agar sesuai grid referensi (ref).
    Hasilnya disimpan ke dst_path.

    Parameters
    ----------
    resampling : str
        'bilinear', 'nearest', 'cubic'
    """
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio diperlukan untuk resampling.")

    resamp_map = {
        "bilinear": Resampling.bilinear,
        "nearest" : Resampling.nearest,
        "cubic"   : Resampling.cubic,
    }
    resamp = resamp_map.get(resampling, Resampling.bilinear)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_path) as ref:
        ref_crs   = ref.crs
        ref_trans = ref.transform
        ref_w     = ref.width
        ref_h     = ref.height

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({
            "crs"       : ref_crs,
            "transform" : ref_trans,
            "width"     : ref_w,
            "height"    : ref_h,
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source      = rasterio.band(src, i),
                    destination = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs     = src.crs,
                    dst_transform = ref_trans,
                    dst_crs     = ref_crs,
                    resampling  = resamp,
                )

    logger.info("Resampling selesai → %s", dst_path.name)
    return dst_path


# ──────────────────────────────────────────────────────────────
# Penggabungan kanal S1 + S2
# ──────────────────────────────────────────────────────────────

def fuse_s1_s2(
    s1_data: np.ndarray,
    s2_data: np.ndarray,
) -> np.ndarray:
    """
    Menggabungkan array S1 dan S2 secara kanal (channel-wise stack).

    Parameters
    ----------
    s1_data : np.ndarray, shape (C1, H, W)
    s2_data : np.ndarray, shape (C2, H, W)

    Returns
    -------
    fused : np.ndarray, shape (C1+C2, H, W)
    """
    if s1_data.shape[1:] != s2_data.shape[1:]:
        raise ValueError(
            f"Dimensi spasial S1 {s1_data.shape[1:]} dan S2 "
            f"{s2_data.shape[1:]} tidak cocok. Lakukan resampling terlebih dahulu."
        )
    return np.concatenate([s1_data, s2_data], axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Simpan / muat konfigurasi preprocessing
# ──────────────────────────────────────────────────────────────

def save_preprocessing_config(config: Dict, path: str | Path) -> None:
    """Menyimpan statistik preprocessing ke JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info("Konfigurasi preprocessing disimpan: %s", path)


def load_preprocessing_config(path: str | Path) -> Dict:
    """Memuat statistik preprocessing dari JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasi preprocessing tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Konfigurasi preprocessing dimuat: %s", path)
    return config


# ──────────────────────────────────────────────────────────────
# Utilitas tambahan
# ──────────────────────────────────────────────────────────────

def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Menghitung NDVI dari kanal Red dan NIR."""
    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Menghitung NDWI dari kanal Green dan NIR."""
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    return ndwi.astype(np.float32)


def check_cloud_coverage(
    data: np.ndarray,
    nodata_value: float = 0.0,
    threshold: float = 0.3,
) -> Tuple[float, bool]:
    """
    Estimasi kasar persentase piksel bermasalah (NoData/cloud proxy).

    Returns
    -------
    ratio   : float  — proporsi piksel tidak valid
    too_cloudy : bool — apakah melebihi threshold
    """
    total = data.size
    if total == 0:
        return 1.0, True
    invalid = np.sum((data == nodata_value) | np.isnan(data) | np.isinf(data))
    ratio = float(invalid / total)
    return ratio, ratio > threshold
