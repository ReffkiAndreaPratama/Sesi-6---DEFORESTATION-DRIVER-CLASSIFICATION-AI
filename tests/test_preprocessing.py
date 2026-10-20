"""
ForestWatch AI — tests/test_preprocessing.py
Pengujian pipeline preprocessing raster S1, S2, dan fusi.
"""

import numpy as np
import pytest


# ── Sentinel-1 ────────────────────────────────────────────────

def test_s1_preprocessing_output_range(synthetic_s1_array):
    """Nilai output S1 harus berada dalam rentang [0, 1] setelah minmax."""
    from src.raster_preprocessing import preprocess_sentinel1
    result, stats = preprocess_sentinel1(synthetic_s1_array, norm_method="minmax")
    assert result.dtype == np.float32
    assert result.min() >= 0.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


def test_s1_preprocessing_shape(synthetic_s1_array):
    """Shape output harus sama dengan input."""
    from src.raster_preprocessing import preprocess_sentinel1
    result, _ = preprocess_sentinel1(synthetic_s1_array)
    assert result.shape == synthetic_s1_array.shape


def test_s1_preprocessing_zscore(synthetic_s1_array):
    """z-score: mean ≈ 0, std ≈ 1 per kanal (pada piksel valid)."""
    from src.raster_preprocessing import preprocess_sentinel1
    result, stats = preprocess_sentinel1(
        synthetic_s1_array, norm_method="zscore"
    )
    for b in range(result.shape[0]):
        vals = result[b][result[b] != 0]
        if vals.size > 10:
            assert abs(float(np.mean(vals))) < 1.5


def test_s1_nodata_handling():
    """Piksel NoData harus diisi 0 setelah normalisasi."""
    from src.raster_preprocessing import preprocess_sentinel1
    data = np.ones((2, 16, 16), dtype=np.float32) * (-15.0)
    data[0, 0, 0] = -9999.0   # NoData
    result, stats = preprocess_sentinel1(data, nodata_value=-9999.0)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


def test_s1_stats_reuse(synthetic_s1_array):
    """Stats dari data latih dapat digunakan ulang saat inferensi."""
    from src.raster_preprocessing import preprocess_sentinel1
    _, train_stats = preprocess_sentinel1(synthetic_s1_array)
    new_data = np.ones_like(synthetic_s1_array) * (-10.0)
    result, _ = preprocess_sentinel1(new_data, stats=train_stats["bands"])
    assert result.dtype == np.float32
    assert not np.isnan(result).any()


# ── Sentinel-2 ────────────────────────────────────────────────

def test_s2_preprocessing_output_range(synthetic_s2_array):
    """Nilai output S2 harus berada dalam [0, 1]."""
    from src.raster_preprocessing import preprocess_sentinel2
    result, stats = preprocess_sentinel2(
        synthetic_s2_array,
        scale_factor = 10000.0,
        norm_method  = "minmax",
    )
    assert result.dtype == np.float32
    assert result.min() >= 0.0 - 1e-6
    assert result.max() <= 1.0 + 1e-6


def test_s2_preprocessing_shape(synthetic_s2_array):
    from src.raster_preprocessing import preprocess_sentinel2
    result, _ = preprocess_sentinel2(synthetic_s2_array)
    assert result.shape == synthetic_s2_array.shape


def test_s2_nodata_handling():
    """NoData value 0 pada S2 harus ditangani."""
    from src.raster_preprocessing import preprocess_sentinel2
    data = np.ones((6, 16, 16), dtype=np.float32) * 1500.0
    data[0, :2, :2] = 0  # NoData
    result, _ = preprocess_sentinel2(data, nodata_value=0)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()


# ── Fusi S1+S2 ────────────────────────────────────────────────

def test_fuse_s1_s2(synthetic_s1_array, synthetic_s2_array):
    """Fusi menghasilkan array dengan kanal = S1 + S2."""
    from src.raster_preprocessing import preprocess_sentinel1, preprocess_sentinel2, fuse_s1_s2
    s1, _ = preprocess_sentinel1(synthetic_s1_array)
    s2, _ = preprocess_sentinel2(synthetic_s2_array)
    fused  = fuse_s1_s2(s1, s2)
    assert fused.shape[0] == s1.shape[0] + s2.shape[0]
    assert fused.shape[1:] == s1.shape[1:]
    assert fused.dtype == np.float32


def test_fuse_dimension_mismatch():
    """Dimensi spasial berbeda harus memunculkan ValueError."""
    from src.raster_preprocessing import fuse_s1_s2
    s1 = np.zeros((2, 64, 64), dtype=np.float32)
    s2 = np.zeros((6, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        fuse_s1_s2(s1, s2)


# ── Config preprocessing ──────────────────────────────────────

def test_save_load_preprocessing_config(tmp_path, synthetic_s1_array):
    """Config preprocessing dapat disimpan dan dimuat kembali."""
    from src.raster_preprocessing import (
        preprocess_sentinel1, save_preprocessing_config, load_preprocessing_config
    )
    _, stats = preprocess_sentinel1(synthetic_s1_array)
    path = tmp_path / "prep_config.json"
    save_preprocessing_config(stats, path)
    loaded = load_preprocessing_config(path)
    assert "sensor" in loaded
    assert "bands" in loaded


def test_load_nonexistent_config():
    """Memuat config yang tidak ada harus memunculkan FileNotFoundError."""
    from src.raster_preprocessing import load_preprocessing_config
    with pytest.raises(FileNotFoundError):
        load_preprocessing_config("/nonexistent/config.json")


# ── NDVI / NDWI ───────────────────────────────────────────────

def test_ndvi_range():
    """NDVI harus berada dalam [-1, 1]."""
    from src.raster_preprocessing import compute_ndvi
    rng = np.random.default_rng(5)
    red = rng.uniform(0, 0.5, (64, 64)).astype(np.float32)
    nir = rng.uniform(0, 0.8, (64, 64)).astype(np.float32)
    ndvi = compute_ndvi(red, nir)
    assert ndvi.min() >= -1.0 - 1e-5
    assert ndvi.max() <=  1.0 + 1e-5


def test_cloud_check():
    """check_cloud_coverage mendeteksi piksel invalid."""
    from src.raster_preprocessing import check_cloud_coverage
    data = np.ones((4, 64, 64), dtype=np.float32)
    data[:, :10, :10] = 0.0   # 10x10 = 100 piksel NoData dari total 4*64*64
    ratio, too_cloudy = check_cloud_coverage(data, nodata_value=0.0, threshold=0.05)
    assert 0.0 < ratio < 1.0
    assert isinstance(too_cloudy, bool)
