# Models — ForestWatch AI

Direktori ini menyimpan file checkpoint model terlatih.

## File Model

| File | Deskripsi | Input Kanal |
|------|-----------|-------------|
| `model_sentinel1.pt` | Model hanya Sentinel-1 (SAR) | 2 (VV, VH) |
| `model_sentinel2.pt` | Model hanya Sentinel-2 (Optik) | 6 (B2,B3,B4,B8,B11,B12) |
| `model_fused.pt`     | Model gabungan S1 + S2 | 8 |

## Format Checkpoint

Setiap file `.pt` menyimpan:
```python
{
    "state_dict"   : model.state_dict(),
    "architecture" : "ForestWatchCNN",
    "in_channels"  : int,
    "num_classes"  : int,
    "metadata"     : {
        "best_epoch"    : int,
        "best_val_loss" : float,
        "best_val_acc"  : float,
    }
}
```

## Melatih Model

```bash
# Mode demo (data sintetis)
python scripts/train_model.py

# Semua mode
python scripts/train_model.py --mode s1
python scripts/train_model.py --mode s2
python scripts/train_model.py --mode fused

# Dengan parameter kustom
python scripts/train_model.py --mode fused --epochs 50 --lr 0.001
```

## Memuat Model di Python

```python
from src.model import load_model
model, metadata = load_model("models/model_fused.pt", device="cpu")
model.eval()
```

## Catatan

- File model besar (> 50 MB) **tidak disertakan** di repositori GitHub
- Latih ulang model menggunakan dataset dan instruksi di atas
- Model yang terlatih pada data sintetis **tidak valid** untuk prediksi nyata
