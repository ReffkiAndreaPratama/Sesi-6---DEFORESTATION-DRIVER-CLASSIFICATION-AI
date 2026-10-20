# 🌿 ForestWatch AI — Bengkulu

**Klasifikasi Penyebab Deforestasi Menggunakan Citra Satelit Sentinel-1 dan Sentinel-2 Berbasis Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Deskripsi Proyek

ForestWatch AI adalah sistem analisis citra satelit berbasis deep learning untuk mengklasifikasikan **penyebab langsung deforestasi** di wilayah Bengkulu, Sumatera. Sistem memanfaatkan citra radar Sentinel-1 dan citra optik Sentinel-2 untuk membedakan antara berbagai penyebab kehilangan tutupan hutan.

### Latar Belakang

Deteksi kehilangan tutupan hutan dari citra satelit sudah dapat dilakukan secara otomatis. Namun, mengidentifikasi **penyebab langsung** perubahan tersebut (apakah akibat pertanian, penebangan, tambang, atau kebakaran) memerlukan analisis yang lebih mendalam. Sistem ini mengembangkan CNN (Convolutional Neural Network) untuk mengklasifikasikan penyebab tersebut berdasarkan karakteristik spektral dan radar.

### Tujuan

1. Mengolah citra satelit Sentinel-1 dan Sentinel-2 untuk analisis deforestasi
2. Mengembangkan model CNN untuk klasifikasi 6 kelas penyebab deforestasi
3. Memanfaatkan informasi komplementer dari SAR dan optik
4. Mengevaluasi kinerja model secara ilmiah dengan metrik yang terstandar
5. Menampilkan hasil dalam bentuk peta geospasial interaktif
6. Menyediakan aplikasi web yang dapat dijalankan secara lokal

---

## 🏷️ Kelas yang Didukung

| Indeks | Kelas Deforestasi |
|--------|-------------------|
| 0 | Pertanian Skala Kecil |
| 1 | Pertanian Skala Besar |
| 2 | Pembangunan Jalan |
| 3 | Penebangan Selektif |
| 4 | Pertambangan |
| 5 | Kebakaran / Gangguan Alami |

> Kelas dapat dikonfigurasi ulang di `config/settings.py` sesuai dataset yang tersedia.

---

## 🛰️ Sumber Dataset

### Sentinel-1 (SAR)
- **Sumber:** [ESA Copernicus Open Access Hub](https://scihub.copernicus.eu) atau [Google Earth Engine](https://earthengine.google.com)
- **Kanal:** VV, VH
- **Resolusi:** 10–20 meter
- **Format:** GeoTIFF, nilai dalam dB

### Sentinel-2 (Optik)
- **Sumber:** [ESA Copernicus Open Access Hub](https://scihub.copernicus.eu)
- **Level:** Level-2A (Bottom-of-Atmosphere)
- **Kanal:** B2, B3, B4, B8, B11, B12
- **Resolusi:** 10–20 meter
- **Format:** GeoTIFF, nilai integer ÷ 10000 = reflektansi

### Label Deforestasi
- Label harus berupa poligon terverifikasi dengan kelas penyebab
- Format: GeoJSON, Shapefile, atau raster label
- Sumber referensi: [Global Forest Watch](https://globalforestwatch.org), survei lapangan

> ⚠️ Periksa lisensi setiap dataset sebelum digunakan untuk penelitian.

---

## 🔧 Metode Preprocessing

1. **Pembacaan raster** dengan rasterio — memuat metadata, CRS, transform
2. **Validasi CRS** — reproyeksi jika S1 dan S2 berbeda sistem koordinat
3. **Penanganan NoData** — masking nilai tidak valid (-9999, 0, NaN)
4. **Normalisasi Sentinel-1:** kliping ke [-50, +10] dB → min-max ke [0, 1]
5. **Normalisasi Sentinel-2:** pembagian scale factor (÷10000) → min-max ke [0, 1]
6. **Fusi kanal** S1 + S2 untuk mode gabungan (8 kanal)
7. **Pembuatan patch** 64×64 piksel dengan sliding window (25% overlap)
8. **Penghubungan label** — modus kelas dalam setiap patch
9. **Pembagian data** — stratified split (train/val/test = 70/15/15)
10. **Simpan konfigurasi** — statistik normalisasi disimpan untuk konsistensi inferensi

---

## 🧠 Arsitektur CNN

### ForestWatchCNN

```
Input (C × 64 × 64)
    ↓
Stem: Conv(3×3) → BN → ReLU
    ↓
Stage 1: 2× ResidualBlock(64ch) → MaxPool
    ↓
Stage 2: Conv + 2× ResidualBlock(128ch) → MaxPool
    ↓
Stage 3: Conv + 2× ResidualBlock(256ch) → MaxPool
    ↓
GlobalAvgPool → Dropout → FC(512) → ReLU → FC(num_classes)
    ↓
Output (num_classes)
```

- **Input channels:** 2 (S1), 6 (S2), atau 8 (Fused)
- **Jumlah parameter:** ~2.5M (base_filters=64)
- **Aktivasi:** ReLU + BatchNorm
- **Regularisasi:** Dropout (0.25), weight decay (1e-4)

### Eksperimen yang Direncanakan

| Mode | Input | Keterangan |
|------|-------|------------|
| S1 Only | 2 kanal | Hanya SAR |
| S2 Only | 6 kanal | Hanya Optik |
| Fused   | 8 kanal | S1 + S2 |

> Perbandingan ketiga mode memerlukan dataset berlabel yang valid. Hasil tidak dapat diklaim sebelum eksperimen benar-benar dijalankan.

---

## 🚀 Instalasi

### Prasyarat
- Python 3.11
- pip
- (Opsional) CUDA untuk GPU

### Langkah Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/username/forestwatch-ai.git
cd forestwatch-ai

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan (Windows)
.venv\Scripts\activate

# 4. Instal dependensi
pip install -r requirements.txt
```

---

## 📂 Persiapan Dataset

### Menggunakan Data Nyata

```bash
# Letakkan file raster ke direktori yang sesuai:
# data/sentinel1/*.tif  → citra SAR
# data/sentinel2/*.tif  → citra optik
# data/labels/          → file label

# Jalankan preprocessing dan pembuatan patch:
python scripts/prepare_data.py
```

### Mode Demo (Tanpa Data Nyata)

```bash
python scripts/prepare_data.py --demo
# Membuat 300 patch sintetis untuk menguji pipeline
# Data sintetis TIDAK mewakili kondisi deforestasi nyata
```

---

## 🏋️ Pelatihan Model

```bash
# Latih model fused (S1 + S2) — default
python scripts/train_model.py

# Latih model spesifik
python scripts/train_model.py --mode s1
python scripts/train_model.py --mode s2
python scripts/train_model.py --mode fused

# Parameter kustom
python scripts/train_model.py --mode fused --epochs 50 --lr 0.001 --batch 16
```

Model terbaik disimpan otomatis di `models/`.

---

## 📊 Evaluasi Model

```bash
# Evaluasi satu model
python scripts/evaluate_model.py --mode fused

# Evaluasi semua mode sekaligus (untuk tabel perbandingan)
python scripts/evaluate_model.py --all
```

Hasil disimpan di `outputs/evaluation/`.

---

## 🔍 Inferensi

```bash
# Inferensi pada file raster
python scripts/predict.py --input data/sentinel2/scene.tif --mode fused

# Mode demo
python scripts/predict.py --demo
```

---

## 🖥️ Menjalankan Aplikasi Web

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

### Menu Aplikasi

| Menu | Fungsi |
|------|--------|
| 🏠 Dashboard | Status sistem, ringkasan model dan dataset |
| 📁 Dataset | Inventori file, distribusi kelas, panduan impor |
| 🔬 Analisis Citra | Muat raster, metadata, visualisasi kanal |
| 🤖 Klasifikasi | Inferensi model, hasil prediksi per patch |
| 📈 Evaluasi | Metrik, confusion matrix, riwayat pelatihan |
| 🗺️ Peta Hasil | Peta interaktif Folium, statistik area |
| ℹ️ Tentang | Dokumentasi proyek, cara penggunaan |

---

## 🗺️ Peta Hasil Klasifikasi

Setelah inferensi berhasil:
1. Buka menu **🗺️ Peta Hasil** di aplikasi
2. Layer hasil klasifikasi ditampilkan di peta Folium interaktif
3. Tooltip per patch menampilkan kelas dan tingkat kepercayaan
4. Ekspor ke GeoTIFF tersedia jika raster memiliki georeferensi valid

---

## 📐 Metrik Evaluasi

| Metrik | Keterangan |
|--------|-----------|
| Accuracy | Proporsi prediksi benar |
| Precision (Macro) | Rata-rata presisi antar kelas |
| Recall (Macro) | Rata-rata recall antar kelas |
| F1-score (Macro) | Rata-rata F1 antar kelas |
| Confusion Matrix | Tabel kesalahan per kelas |
| Inferensi (ms) | Waktu rata-rata per sampel |

---

## 📏 Hasil Eksperimen

> Bagian ini akan diisi setelah eksperimen dengan dataset nyata selesai dilakukan.
> Jangan membandingkan model yang dilatih pada data berbeda.

---

## ⚠️ Keterbatasan Sistem

1. **Ketergantungan pada label yang valid** — model tanpa label nyata tidak dapat mengklasifikasikan deforestasi dengan andal
2. **Data sintetis hanya untuk pengujian** — bukan representasi kondisi lapangan
3. **Validasi lapangan diperlukan** — prediksi model harus diverifikasi dengan data survei
4. **Pengaruh awan** — citra Sentinel-2 berawan dapat menurunkan kualitas prediksi
5. **Resolusi spasial** — detail halus di bawah 10 meter tidak dapat dideteksi
6. **Generalisasi** — model yang dilatih di satu wilayah mungkin tidak berlaku di wilayah lain

---

## 📁 Struktur Proyek

```
forestwatch-ai/
├── app.py                    ← Aplikasi Streamlit
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── settings.py           ← Konfigurasi global
├── src/
│   ├── data_loader.py        ← Pembacaan raster
│   ├── raster_preprocessing.py ← Pipeline preprocessing
│   ├── patch_generator.py    ← Pembuatan patch
│   ├── dataset.py            ← PyTorch Dataset
│   ├── model.py              ← Arsitektur CNN
│   ├── train.py              ← Loop pelatihan
│   ├── inference.py          ← Inferensi
│   ├── evaluation.py         ← Metrik evaluasi
│   ├── geospatial_utils.py   ← Utilitas GIS
│   └── visualization.py      ← Grafik & peta
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_evaluation.py
├── data/
│   ├── sentinel1/
│   ├── sentinel2/
│   └── labels/
├── models/
└── outputs/
```

---

## 🔬 Pengujian

```bash
# Jalankan semua tests
pytest tests/ -v

# Dengan laporan cakupan
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📚 Referensi

- Hansen, M.C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*, 342(6160), 850–853.
- Torres, R., et al. (2012). GMES Sentinel-1 mission. *Remote Sensing of Environment*, 120, 9–24.
- Drusch, M., et al. (2012). Sentinel-2: ESA's Optical High-Resolution Mission for GMES. *Remote Sensing of Environment*, 120, 25–36.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

---

## 📄 Lisensi

MIT License — Bebas digunakan untuk keperluan penelitian dan pendidikan.

---

## 👤 Pengembang

Dikembangkan sebagai proyek penelitian skripsi/tesis di bidang remote sensing dan deep learning.  
Wilayah studi: **Bengkulu, Sumatera, Indonesia**
