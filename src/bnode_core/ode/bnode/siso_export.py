from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

SISO_DIMENSIONS_FILE_NAME = "siso_dimensions.json"


class SISOWrapper(nn.Module):
    """Wraps a multi-input/output module into a single-input single-output interface.

    Slices the concatenated input vector into named keyword arguments for the wrapped
    module, then concatenates any tuple output back into a single tensor.
    """

    def __init__(self, module: nn.Module, input_specs: list[tuple[str, int]]) -> None:
        super().__init__()
        self.module = module
        self.input_specs = input_specs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = 0
        kwargs: dict[str, torch.Tensor] = {}
        for name, dim in self.input_specs:
            kwargs[name] = x[:, offset : offset + dim]
            offset += dim
        result = self.module(**kwargs)
        if isinstance(result, tuple):
            return torch.cat(list(result), dim=1)
        return result


def build_input_specs(inputs: dict[str, torch.Tensor]) -> list[dict]:
    specs, offset = [], 0
    for name, tensor in inputs.items():
        dim = tensor.shape[1]
        specs.append({"name": name, "dim": dim, "start": offset, "end": offset + dim})
        offset += dim
    return specs


def build_output_specs(result: torch.Tensor | tuple, output_names: list[str]) -> list[dict]:
    tensors = list(result) if isinstance(result, tuple) else [result]
    specs, offset = [], 0
    for name, tensor in zip(output_names, tensors):
        dim = tensor.shape[1]
        specs.append({"name": name, "dim": dim, "start": offset, "end": offset + dim})
        offset += dim
    return specs


def write_siso_dimensions(module_specs: dict, path: Path) -> None:
    """Write siso_dimensions.json with all module input/output specs."""
    path.write_text(json.dumps({"version": 1, **module_specs}, indent=2))
