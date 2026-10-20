"""
ForestWatch AI — tests/conftest.py
Fixture pytest yang digunakan bersama oleh semua test module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Pastikan root proyek ada di sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Dimensi standar fixture ────────────────────────────────────
PATCH_SIZE   = 32      # kecil agar test cepat
NUM_BANDS_S1 = 2
NUM_BANDS_S2 = 6
NUM_BANDS_FUSED = NUM_BANDS_S1 + NUM_BANDS_S2
NUM_CLASSES  = 6
H, W         = 128, 128   # dimensi raster sintetis


@pytest.fixture(scope="session")
def synthetic_s1_array():
    """Array SAR Sentinel-1 sintetis (2, H, W)."""
    rng = np.random.default_rng(0)
    return rng.uniform(-25.0, -5.0, (NUM_BANDS_S1, H, W)).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_s2_array():
    """Array optik Sentinel-2 sintetis (6, H, W) — skala integer BOA."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 3000, (NUM_BANDS_S2, H, W)).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_fused_array():
    """Array gabungan S1+S2 sintetis (8, H, W), sudah ternormalisasi [0,1]."""
    rng = np.random.default_rng(2)
    return rng.uniform(0.0, 1.0, (NUM_BANDS_FUSED, H, W)).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_label_raster():
    """Raster label integer (H, W), nilai 1..NUM_CLASSES."""
    rng = np.random.default_rng(3)
    return rng.integers(1, NUM_CLASSES + 1, (H, W)).astype(np.int32)


@pytest.fixture(scope="session")
def synthetic_patches():
    """Daftar patch sintetis berlabel dari patch_generator."""
    from src.patch_generator import generate_synthetic_patches
    return generate_synthetic_patches(
        num_patches=60,
        num_classes=NUM_CLASSES,
        channels=NUM_BANDS_FUSED,
        patch_size=PATCH_SIZE,
        seed=42,
    )


@pytest.fixture(scope="session")
def tmp_geotiff(tmp_path_factory):
    """
    File GeoTIFF sintetis sementara (digunakan untuk test data_loader).
    Dikembalikan sebagai Path.
    """
    try:
        from src.data_loader import generate_synthetic_raster
        tmp_dir  = tmp_path_factory.mktemp("rasters")
        tif_path = tmp_dir / "test_s1.tif"
        generate_synthetic_raster(
            path        = tif_path,
            num_bands   = NUM_BANDS_S1,
            height      = H,
            width       = W,
            value_range = (-25.0, -5.0),
            seed        = 99,
        )
        return tif_path
    except Exception:
        pytest.skip("rasterio tidak tersedia — test raster dilewati.")
