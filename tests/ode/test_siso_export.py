"""Unit tests for the SISOWrapper and related helpers in siso_export.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn

from bnode_core.ode.bnode.siso_export import (
    SISOWrapper,
    build_input_specs,
    build_output_specs,
    write_siso_dimensions,
    SISO_DIMENSIONS_FILE_NAME,
)


class _TwoInputTwoOutputModule(nn.Module):
    """Toy module: two inputs, two outputs (their squares)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return a ** 2, b ** 2


class _SingleInputModule(nn.Module):
    """Toy module: one input, one output (doubled)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def test_siso_wrapper_forward_equivalence():
    mod = _TwoInputTwoOutputModule()
    a = torch.randn(2, 3)
    b = torch.randn(2, 4)
    x = torch.cat([a, b], dim=1)

    wrapper = SISOWrapper(mod, [("a", 3), ("b", 4)])
    out = wrapper(x)

    expected = torch.cat([a ** 2, b ** 2], dim=1)
    assert out.shape == (2, 7)
    assert torch.allclose(out, expected)


def test_siso_wrapper_single_output():
    mod = _SingleInputModule()
    x = torch.randn(2, 5)
    wrapper = SISOWrapper(mod, [("x", 5)])
    out = wrapper(x)
    assert torch.allclose(out, x * 2)


def test_siso_wrapper_onnx_export_produces_single_input_output():
    mod = _TwoInputTwoOutputModule()
    wrapper = SISOWrapper(mod, [("a", 3), ("b", 4)])
    x = torch.randn(1, 7)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = Path(f.name)

    torch.onnx.export(
        wrapper,
        args=(x,),
        f=str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    model = onnx.load(str(path))
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"
    path.unlink()


def test_build_input_specs():
    inputs = {
        "a": torch.zeros(1, 3),
        "b": torch.zeros(1, 4),
    }
    specs = build_input_specs(inputs)
    assert specs == [
        {"name": "a", "dim": 3, "start": 0, "end": 3},
        {"name": "b", "dim": 4, "start": 3, "end": 7},
    ]


def test_build_output_specs_tuple():
    result = (torch.zeros(1, 3), torch.zeros(1, 4))
    specs = build_output_specs(result, ["mu", "logvar"])
    assert specs == [
        {"name": "mu", "dim": 3, "start": 0, "end": 3},
        {"name": "logvar", "dim": 4, "start": 3, "end": 7},
    ]


def test_write_siso_dimensions(tmp_path: Path):
    module_specs = {
        "encoder_states": {
            "siso_onnx": "encoder_states_siso.onnx",
            "input": [{"name": "x", "dim": 5, "start": 0, "end": 5}],
            "output": [{"name": "latent_states_mu", "dim": 3, "start": 0, "end": 3}],
            "normalization_mu": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    }
    out_path = tmp_path / SISO_DIMENSIONS_FILE_NAME
    write_siso_dimensions(module_specs, out_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["version"] == 1
    assert "encoder_states" in data
    assert data["encoder_states"]["normalization_mu"] == [0.1, 0.2, 0.3, 0.4, 0.5]
