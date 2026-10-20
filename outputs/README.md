# Outputs — ForestWatch AI

Direktori ini menyimpan semua hasil komputasi yang dihasilkan otomatis.

## Struktur

```
outputs/
├── patches/           ← Patch .npy hasil prepare_data.py
│   ├── train/
│   ├── val/
│   └── test/
├── plots/             ← Grafik dan riwayat pelatihan
│   └── history_*.csv
├── predictions/       ← Hasil prediksi inferensi
│   ├── *_predictions.csv
│   └── *_prediction_map.tif
├── evaluation/        ← Hasil evaluasi model
│   ├── eval_s1.json
│   ├── eval_s2.json
│   ├── eval_fused.json
│   └── comparison_table.json
├── exports/           ← Ekspor GeoJSON/GeoTIFF
├── manifest.csv       ← Daftar semua patch
└── prep_config.json   ← Konfigurasi preprocessing
```

## File Penting

- **manifest.csv** — inventori semua patch beserta label dan path file
- **prep_config.json** — statistik normalisasi yang harus digunakan saat inferensi
- **comparison_table.json** — tabel perbandingan metrik antar model

## Catatan

- Direktori `patches/` bisa berukuran besar — tidak disertakan di GitHub
- File `prep_config.json` **wajib ada** saat menjalankan inferensi agar normalisasi konsisten
