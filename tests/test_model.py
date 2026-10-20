"""
ForestWatch AI — tests/test_model.py
Pengujian arsitektur CNN, forward pass, simpan/muat model.
"""

import numpy as np
import pytest
import torch


NUM_CLASSES = 6
PATCH_SIZE  = 32


@pytest.fixture
def cnn_model():
    from src.model import ForestWatchCNN
    return ForestWatchCNN(in_channels=8, num_classes=NUM_CLASSES, base_filters=16)


@pytest.fixture
def s1_model():
    from src.model import ForestWatchCNN
    return ForestWatchCNN(in_channels=2, num_classes=NUM_CLASSES, base_filters=16)


# ── Forward pass ──────────────────────────────────────────────

def test_forward_pass_fused(cnn_model):
    """Forward pass dengan input fused (8 kanal) menghasilkan logit yang benar."""
    batch   = torch.randn(4, 8, PATCH_SIZE, PATCH_SIZE)
    output  = cnn_model(batch)
    assert output.shape == (4, NUM_CLASSES)
    assert not torch.isnan(output).any()


def test_forward_pass_s1(s1_model):
    """Forward pass dengan input S1 (2 kanal)."""
    batch  = torch.randn(2, 2, PATCH_SIZE, PATCH_SIZE)
    output = s1_model(batch)
    assert output.shape == (2, NUM_CLASSES)


def test_forward_pass_s2():
    """Forward pass dengan input S2 (6 kanal)."""
    from src.model import ForestWatchCNN
    model  = ForestWatchCNN(in_channels=6, num_classes=NUM_CLASSES, base_filters=16)
    batch  = torch.randn(3, 6, PATCH_SIZE, PATCH_SIZE)
    output = model(batch)
    assert output.shape == (3, NUM_CLASSES)


def test_predict_proba_sums_to_one(cnn_model):
    """Probabilitas harus berjumlah 1 per sampel."""
    batch = torch.randn(4, 8, PATCH_SIZE, PATCH_SIZE)
    probs = cnn_model.predict_proba(batch)
    sums  = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_get_features(cnn_model):
    """get_features mengembalikan vektor fitur (sebelum head)."""
    batch    = torch.randn(2, 8, PATCH_SIZE, PATCH_SIZE)
    features = cnn_model.get_features(batch)
    assert features.ndim == 2
    assert features.shape[0] == 2


def test_count_parameters(cnn_model):
    """Parameter model harus > 0."""
    n = cnn_model.count_parameters()
    assert n > 0


def test_single_class():
    """Model dengan 1 kelas tidak crash."""
    from src.model import ForestWatchCNN
    model  = ForestWatchCNN(in_channels=8, num_classes=1, base_filters=8)
    batch  = torch.randn(2, 8, PATCH_SIZE, PATCH_SIZE)
    output = model(batch)
    assert output.shape == (2, 1)


# ── build_model factory ───────────────────────────────────────

def test_build_model_cnn():
    from src.model import build_model
    model = build_model("cnn", in_channels=8, num_classes=6, base_filters=16)
    assert model.__class__.__name__ == "ForestWatchCNN"


def test_build_model_invalid_arch():
    from src.model import build_model
    with pytest.raises(ValueError):
        build_model("transformer", in_channels=8, num_classes=6)


# ── Simpan / muat model ───────────────────────────────────────

def test_save_load_model(tmp_path, cnn_model):
    """Model dapat disimpan dan dimuat kembali dengan state_dict identik."""
    from src.model import save_model, load_model
    path = tmp_path / "test_model.pt"
    save_model(cnn_model, path, metadata={"test": True})

    loaded_model, meta = load_model(str(path), device="cpu")
    assert meta.get("test") is True

    # Bandingkan parameter
    for (k1, v1), (k2, v2) in zip(
        cnn_model.state_dict().items(),
        loaded_model.state_dict().items()
    ):
        assert torch.allclose(v1, v2), f"Parameter {k1} berbeda setelah load."


def test_load_nonexistent_model():
    from src.model import load_model
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/model.pt")


def test_model_eval_mode(tmp_path, cnn_model):
    """Model dimuat dalam mode eval (tidak training)."""
    from src.model import save_model, load_model
    path = tmp_path / "eval_test.pt"
    save_model(cnn_model, path)
    loaded, _ = load_model(str(path), device="cpu")
    assert not loaded.training


# ── get_device ────────────────────────────────────────────────

def test_get_device_cpu():
    from src.model import get_device
    device = get_device("cpu")
    assert str(device) == "cpu"


def test_get_device_auto():
    from src.model import get_device
    device = get_device("auto")
    assert device.type in ("cpu", "cuda")
