"""
Pytest unit tests for Intern 2 — Echo + Base Model Fusion.
Run: pytest tests/test_intern2.py -v
"""

import torch
import pytest
from src.intern2_echo.dataset import ef_to_severity, read_ed_es
from src.intern2_echo.model import EchoEncoder, EchoBranch, CardiacBaseModel


# ── ef_to_severity ─────────────────────────────────────────────────────────────

def test_ef_normal():
    assert ef_to_severity(65.0) == 0

def test_ef_normal_boundary():
    assert ef_to_severity(50.0) == 0

def test_ef_mild():
    assert ef_to_severity(45.0) == 1

def test_ef_mild_boundary():
    assert ef_to_severity(40.0) == 1

def test_ef_severe():
    assert ef_to_severity(35.0) == 2

def test_ef_severe_boundary():
    assert ef_to_severity(39.9) == 2


# ── EchoEncoder ────────────────────────────────────────────────────────────────

@pytest.fixture
def echo_encoder():
    return EchoEncoder(pretrained=False)

def test_echo_encoder_output_shape(echo_encoder):
    x = torch.randn(2, 3, 224, 224)
    out = echo_encoder(x)
    assert out.shape == (2, 512), f"Expected (2,512), got {out.shape}"

def test_echo_encoder_no_nan(echo_encoder):
    x = torch.randn(2, 3, 224, 224)
    out = echo_encoder(x)
    assert not torch.isnan(out).any()


# ── EchoBranch ─────────────────────────────────────────────────────────────────

@pytest.fixture
def echo_branch():
    return EchoBranch(pretrained=False)

def test_echo_branch_output_shape(echo_branch):
    ed_es = torch.randn(2, 2, 3, 224, 224)
    out = echo_branch(ed_es)
    assert out.shape == (2, 512)


# ── CardiacBaseModel (mock ECG encoder) ────────────────────────────────────────

class MockECGEncoder(torch.nn.Module):
    """Simulates Intern 1's ResNet1D during testing — no real weights needed."""
    def forward(self, x, return_features=False):
        feat = torch.zeros(x.size(0), 512)
        logits = torch.zeros(x.size(0), 3)
        return (logits, feat) if return_features else logits

@pytest.fixture
def base_model():
    ecg_enc = MockECGEncoder()
    echo_branch = EchoBranch(pretrained=False)
    return CardiacBaseModel(ecg_enc, echo_branch)

def test_base_model_output_shapes(base_model):
    ecg = torch.randn(2, 12, 5000)
    ed_es = torch.randn(2, 2, 3, 224, 224)
    logits, score = base_model(ecg, ed_es)
    assert logits.shape == (2, 3)
    assert score.shape == (2,)

def test_base_model_score_range(base_model):
    ecg = torch.randn(2, 12, 5000)
    ed_es = torch.randn(2, 2, 3, 224, 224)
    _, score = base_model(ecg, ed_es)
    assert (score >= 0).all() and (score <= 1).all(), "Score must be in [0,1]"

def test_base_model_return_features(base_model):
    ecg = torch.randn(2, 12, 5000)
    ed_es = torch.randn(2, 2, 3, 224, 224)
    logits, score, feat = base_model(ecg, ed_es, return_features=True)
    assert feat.shape == (2, 1024), f"Expected (2,1024), got {feat.shape}"
