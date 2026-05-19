from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


RESTART_CHECKPOINT_FILENAME = "training_restart_checkpoint.pt"
TRAINING_COMPLETE_MARKER_FILENAME = "training_complete.marker"


class CheckpointRequestedExit(RuntimeError):
    """Raised when training should stop after persisting a restart checkpoint.
    This class is used inside testing the restart.
    """

"""Save functions here, that can be used in classes below and elsewhere"""

def _pickle_value(value: Any) -> torch.Tensor:
    """Pickle any value into a single uint8 tensor."""
    data = bytearray(pickle.dumps(value))
    return torch.frombuffer(data, dtype=torch.uint8).clone()


def _unpickle_value(encoded: torch.Tensor) -> Any:
    """Reconstruct a value from a pickled uint8 tensor."""
    return pickle.loads(bytes(encoded.cpu().tolist()))  # noqa: S301


def capture_rng_state(use_cuda: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if use_cuda and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], use_cuda: bool) -> None:
    if not state:
        return
    torch.random.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if use_cuda and torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _pickle_rng_state(state: dict[str, Any]) -> torch.Tensor:
    """Pickle the heterogeneous RNG state into a single uint8 tensor."""
    if not state:
        return torch.zeros(0, dtype=torch.uint8)
    data = bytearray(pickle.dumps(state))
    return torch.frombuffer(data, dtype=torch.uint8).clone()


def _unpickle_rng_state(encoded: torch.Tensor) -> dict[str, Any]:
    """Reconstruct the RNG state from a pickled uint8 tensor."""
    if encoded.numel() == 0:
        return {}
    return pickle.loads(bytes(encoded.cpu().tolist()))  # noqa: S301


def _encode_nullable_int(value: int | None) -> int:
    return -1 if value is None else int(value)


def _decode_nullable_int(value: int) -> int | None:
    return None if value < 0 else value


# ---------------------------------------------------------------------------
# FIELD_REGISTRY helpers — used by both state classes
# ---------------------------------------------------------------------------

_FieldRegistry = list[tuple[str, str, torch.dtype | None, Any, Any]]


def _registry_register_buffers(module: torch.nn.Module, registry: _FieldRegistry) -> None:
    """Call register_buffer for every entry in a FIELD_REGISTRY."""
    for attr, buf, dtype, encode, _ in registry:
        value = getattr(module, attr)
        if dtype is not None:
            module.register_buffer(buf, torch.tensor(encode(value), dtype=dtype))
        else:
            module.register_buffer(buf, encode(value))


def _registry_encode_to_buffers(module: torch.nn.Module, registry: _FieldRegistry) -> None:
    """Encode all Python attributes to their buffers before torch.save."""
    for attr, buf, dtype, encode, _ in registry:
        value = getattr(module, attr)
        if dtype is not None:
            setattr(module, buf, torch.tensor(encode(value), dtype=dtype))
        else:
            setattr(module, buf, encode(value))


def _registry_decode_from_buffers(module: torch.nn.Module, registry: _FieldRegistry) -> None:
    """Restore Python attributes from their buffers after load_state_dict."""
    for attr, buf, dtype, _, decode in registry:
        buffer = getattr(module, buf)
        setattr(module, attr, decode(buffer.item()) if dtype is not None else decode(buffer))


def _registry_preload_variable_buffers(
    module: torch.nn.Module,
    registry: _FieldRegistry,
    state_dict: dict[str, Any],
) -> None:
    """Pre-size variable-length pickle buffers before load_state_dict(strict=True)."""
    for _, buf, dtype, _, _ in registry:
        if dtype is None:
            setattr(module, buf, state_dict[buf])


class TrainAllPhasesState(torch.nn.Module):
    """Minimal persisted state for ``train_all_phases()``."""

    STATE_VERSION = 3

    # (attr_name, buffer_name, dtype_or_None, encode_fn, decode_fn)
    # dtype=None means a variable-size pickle tensor; dtype set means a fixed torch.tensor.
    FIELD_REGISTRY = [
        ("job_idx",           "_job_idx",             torch.int64, int,           int),
        ("next_epoch_anchor", "_next_epoch_anchor",   torch.int64, int,           int),
        ("mlflow_run_id",     "_mlflow_run_id_bytes", None,        _pickle_value, _unpickle_value),
    ]

    def __init__(
        self,
        *,
        job_idx: int = 0,
        next_epoch_anchor: int = 0,
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.job_idx = job_idx
        self.next_epoch_anchor = next_epoch_anchor
        self.mlflow_run_id = mlflow_run_id
        self.register_buffer("_state_version", torch.tensor(self.STATE_VERSION, dtype=torch.int64))
        _registry_register_buffers(self, self.FIELD_REGISTRY)

    def to_state_dict(self) -> dict:
        self._state_version = torch.tensor(self.STATE_VERSION, dtype=torch.int64)
        _registry_encode_to_buffers(self, self.FIELD_REGISTRY)
        return self.state_dict()

    def load_from_state_dict(self, state_dict: dict) -> "TrainAllPhasesState":
        self._validate_state_version(state_dict)
        _registry_preload_variable_buffers(self, self.FIELD_REGISTRY, state_dict)
        self.load_state_dict(state_dict, strict=True)
        _registry_decode_from_buffers(self, self.FIELD_REGISTRY)
        return self

    def save(self, path: Path) -> None:
        sd = self.to_state_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, path)
        logging.info("Saved train_all_phases_state to %s", path)

    def load(self, path: Path) -> "TrainAllPhasesState":
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        return self.load_from_state_dict(state_dict)

    def _validate_state_version(self, state_dict: dict[str, Any]) -> None:
        if "_state_version" not in state_dict:
            raise ValueError(
                "TrainAllPhasesState checkpoint is missing '_state_version'."
            )
        saved_version = int(state_dict["_state_version"].item())
        if saved_version != self.STATE_VERSION:
            raise ValueError(
                "TrainAllPhasesState checkpoint version mismatch: "
                f"saved={saved_version}, expected={self.STATE_VERSION}."
            )


class TrainOnePhaseState(torch.nn.Module):
    """Minimal persisted state for ``train_one_phase()``."""

    STATE_VERSION = 4

    # (attr_name, buffer_name, dtype_or_None, encode_fn, decode_fn)
    # dtype=None means a variable-size pickle tensor; dtype set means a fixed torch.tensor.
    FIELD_REGISTRY = [
        ("phase_epoch",                    "_phase_epoch",                    torch.int64, int,                  int),
        ("nan_counter",                    "_nan_counter",                    torch.int64, int,                  int),
        ("grad_norm_last_reduced_counter", "_grad_norm_last_reduced_counter", torch.int64, int,                  int),
        ("stable_epochs",                  "_stable_epochs",                  torch.int64, int,                  int),
        ("deterministic_mode_active",      "_deterministic_mode_active",      torch.bool,  bool,                 bool),
        ("seq_len_increase_in_batches",    "_seq_len_increase_in_batches",    torch.int64, _encode_nullable_int, _decode_nullable_int),
        ("rng_state",                      "_rng_state_bytes",                None,        _pickle_rng_state,    _unpickle_rng_state),
    ]

    def __init__(
        self,
        *,
        phase_epoch: int = 0,
        nan_counter: int = 0,
        grad_norm_last_reduced_counter: int = 0,
        stable_epochs: int = 0,
        deterministic_mode_active: bool = False,
        seq_len_increase_in_batches: int | None = None,
        rng_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.phase_epoch = phase_epoch
        self.nan_counter = nan_counter
        self.grad_norm_last_reduced_counter = grad_norm_last_reduced_counter
        self.stable_epochs = stable_epochs
        self.deterministic_mode_active = deterministic_mode_active
        self.seq_len_increase_in_batches = seq_len_increase_in_batches
        self.rng_state = rng_state or {}
        self.register_buffer("_state_version", torch.tensor(self.STATE_VERSION, dtype=torch.int64))
        _registry_register_buffers(self, self.FIELD_REGISTRY)

    def to_state_dict(self) -> dict:
        self._state_version = torch.tensor(self.STATE_VERSION, dtype=torch.int64)
        _registry_encode_to_buffers(self, self.FIELD_REGISTRY)
        return self.state_dict()

    def load_from_state_dict(self, state_dict: dict) -> "TrainOnePhaseState":
        state_dict = self._filter_state_dict_for_registered_modules(state_dict)
        self._validate_state_version(state_dict)
        _registry_preload_variable_buffers(self, self.FIELD_REGISTRY, state_dict)
        self.load_state_dict(state_dict, strict=True)
        _registry_decode_from_buffers(self, self.FIELD_REGISTRY)
        return self

    def save(self, path: Path) -> None:
        sd = self.to_state_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, path)
        logging.info("Saved train_one_phase_state to %s", path)

    def load(self, path: Path) -> "TrainOnePhaseState":
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        return self.load_from_state_dict(state_dict)

    def _validate_state_version(self, state_dict: dict[str, Any]) -> None:
        if "_state_version" not in state_dict:
            raise ValueError(
                "TrainOnePhaseState checkpoint is missing '_state_version'."
            )
        saved_version = int(state_dict["_state_version"].item())
        if saved_version != self.STATE_VERSION:
            raise ValueError(
                "TrainOnePhaseState checkpoint version mismatch: "
                f"saved={saved_version}, expected={self.STATE_VERSION}."
            )

    def _filter_state_dict_for_registered_modules(
        self,
        state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        has_early_stopping_module = "early_stopping" in self._modules
        has_early_stopping_state = any(
            key.startswith("early_stopping.") for key in state_dict
        )
        if has_early_stopping_state and not has_early_stopping_module:
            return {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("early_stopping.")
            }
        return state_dict


def load_train_all_phases_state_metadata(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    outer = TrainAllPhasesState().load_from_state_dict(bundle["outer"])
    return {
        "job_idx": outer.job_idx,
        "next_epoch_anchor": outer.next_epoch_anchor,
        "mlflow_run_id": outer.mlflow_run_id,
        "restart_state_path": str(path.resolve()),
        "hydra_output_dir": str(path.resolve().parent.resolve()),
    }
