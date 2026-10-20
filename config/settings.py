"""
ForestWatch AI — Konfigurasi Global
Semua parameter proyek dikonfigurasi di sini agar konsisten
antara preprocessing, pelatihan, inferensi, dan aplikasi.
"""

from pathlib import Path

# ─── Root Proyek ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Direktori Data ───────────────────────────────────────────
DATA_DIR         = ROOT_DIR / "data"
SENTINEL1_DIR    = DATA_DIR / "sentinel1"
SENTINEL2_DIR    = DATA_DIR / "sentinel2"
LABELS_DIR       = DATA_DIR / "labels"

# ─── Direktori Output ─────────────────────────────────────────
MODELS_DIR       = ROOT_DIR / "models"
OUTPUTS_DIR      = ROOT_DIR / "outputs"
PREDICTIONS_DIR  = OUTPUTS_DIR / "predictions"
PLOTS_DIR        = OUTPUTS_DIR / "plots"
EXPORTS_DIR      = OUTPUTS_DIR / "exports"

# Pastikan semua direktori output ada
for _d in [MODELS_DIR, PREDICTIONS_DIR, PLOTS_DIR, EXPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Kanal Sentinel-1 ─────────────────────────────────────────
# VV: polarisasi vertikal-vertikal, VH: vertikal-horizontal
S1_BANDS         = ["VV", "VH"]
S1_NODATA        = -9999.0
# Rentang nilai dB yang wajar untuk SAR (sebelum normalisasi)
S1_VALID_RANGE   = (-50.0, 10.0)

# ─── Kanal Sentinel-2 ─────────────────────────────────────────
# Band yang digunakan: B2 Blue, B3 Green, B4 Red, B8 NIR,
#                      B11 SWIR1, B12 SWIR2
S2_BANDS         = ["B2", "B3", "B4", "B8", "B11", "B12"]
S2_NODATA        = 0
# Nilai reflektansi setelah konversi (Level-2A, range 0–1)
S2_VALID_RANGE   = (0.0, 1.0)
# Faktor skala jika nilai asli dalam integer (misalnya ESA BOA = /10000)
S2_SCALE_FACTOR  = 10000.0

# ─── Kelas Deforestasi ────────────────────────────────────────
# Sesuaikan dengan label dataset yang benar-benar tersedia.
# Kunci = nama kelas (string), Nilai = indeks integer
DEFORESTATION_CLASSES = {
    "Pertanian Skala Kecil"  : 0,
    "Pertanian Skala Besar"  : 1,
    "Pembangunan Jalan"      : 2,
    "Penebangan Selektif"    : 3,
    "Pertambangan"           : 4,
    "Kebakaran / Gangguan Alami": 5,
}

# Palet warna per kelas (RGB hex) untuk visualisasi
CLASS_COLORS = {
    "Pertanian Skala Kecil"     : "#f59e0b",
    "Pertanian Skala Besar"     : "#ef4444",
    "Pembangunan Jalan"         : "#6b7280",
    "Penebangan Selektif"       : "#84cc16",
    "Pertambangan"              : "#8b5cf6",
    "Kebakaran / Gangguan Alami": "#f97316",
}

NUM_CLASSES = len(DEFORESTATION_CLASSES)
CLASS_NAMES = list(DEFORESTATION_CLASSES.keys())

# ─── Parameter Patch ──────────────────────────────────────────
PATCH_SIZE       = 64        # Piksel (lebar × tinggi)
PATCH_OVERLAP    = 0.25      # Fraksi overlap antar patch (0–<1)
MIN_VALID_RATIO  = 0.8       # Minimal piksel valid dalam satu patch

# ─── Jumlah Kanal Model ───────────────────────────────────────
# Dihitung otomatis; bisa di-override setelah mengetahui data aktual
NUM_CHANNELS_S1   = len(S1_BANDS)       # 2
NUM_CHANNELS_S2   = len(S2_BANDS)       # 6
NUM_CHANNELS_FUSED = NUM_CHANNELS_S1 + NUM_CHANNELS_S2  # 8

# ─── Hyperparameter Pelatihan ─────────────────────────────────
BATCH_SIZE        = 16
NUM_EPOCHS        = 50
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4
LR_STEP_SIZE      = 15       # Setiap N epoch, LR dikalikan LR_GAMMA
LR_GAMMA          = 0.5
EARLY_STOP_PATIENCE = 10     # Epoch tanpa peningkatan sebelum berhenti
VAL_SPLIT         = 0.15     # Proporsi validasi dari data latih
TEST_SPLIT        = 0.15     # Proporsi pengujian dari seluruh data

# ─── Nama File Model ──────────────────────────────────────────
MODEL_FILENAME_S1    = "model_sentinel1.pt"
MODEL_FILENAME_S2    = "model_sentinel2.pt"
MODEL_FILENAME_FUSED = "model_fused.pt"

# ─── Perangkat Komputasi ──────────────────────────────────────
# "auto" → gunakan CUDA jika tersedia, fallback ke CPU
DEVICE           = "auto"

# ─── Seed Reproduksibilitas ───────────────────────────────────
RANDOM_SEED      = 42

# ─── Logging ──────────────────────────────────────────────────
LOG_DIR          = ROOT_DIR / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Format Ekspor ────────────────────────────────────────────
EXPORT_CRS       = "EPSG:4326"   # WGS 84

# ─── Mode Demonstrasi ─────────────────────────────────────────
# True  → gunakan data sintetis saat dataset belum tersedia
# False → hanya data nyata
DEMO_MODE        = True

# Dimensi citra sintetis untuk demo dan pengujian
DEMO_IMAGE_SIZE  = (256, 256)   # (tinggi, lebar) piksel
