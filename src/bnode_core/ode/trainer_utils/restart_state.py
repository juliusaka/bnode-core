from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


OUTER_RESTART_STATE_FILENAME = "training_outer_restart.pt"
INNER_RESTART_STATE_FILENAME = "training_inner_restart.pt"


class CheckpointRequestedExit(RuntimeError):
    """Raised when training should stop after persisting a restart checkpoint."""


def _encode_optional_string(value: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    if value is None:
        return torch.zeros(0, dtype=torch.uint8), torch.tensor(True, dtype=torch.bool)
    return (
        torch.tensor(list(value.encode("utf-8")), dtype=torch.uint8),
        torch.tensor(False, dtype=torch.bool),
    )


def _decode_optional_string(value_bytes: torch.Tensor, is_none: torch.Tensor) -> str | None:
    if bool(is_none.item()):
        return None
    return bytes(int(x) for x in value_bytes.tolist()).decode("utf-8")


def _resolve_serializer(instance, field_name: str, serializer_name: str):
    serializer = getattr(instance, serializer_name, None)
    if serializer is None or not callable(serializer):
        raise ValueError(
            f"{instance.__class__.__name__}.{field_name} requires serializer "
            f"method '{serializer_name}', but it is missing."
        )
    return serializer


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


def _encode_rng_state(
    state: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not state:
        return {
            "torch_cpu": torch.zeros(0, dtype=torch.uint8),
            "torch_cuda_flat": torch.zeros(0, dtype=torch.uint8),
            "torch_cuda_lengths": torch.zeros(0, dtype=torch.int64),
            "numpy_alg": torch.zeros(0, dtype=torch.uint8),
            "numpy_state": torch.zeros(0, dtype=torch.int64),
            "numpy_pos": torch.tensor(0, dtype=torch.int64),
            "numpy_has_gauss": torch.tensor(False, dtype=torch.bool),
            "numpy_cached_gaussian": torch.tensor(0.0, dtype=torch.float64),
            "python_version": torch.tensor(0, dtype=torch.int64),
            "python_state": torch.zeros(0, dtype=torch.int64),
            "python_has_gauss": torch.tensor(False, dtype=torch.bool),
            "python_cached_gaussian": torch.tensor(0.0, dtype=torch.float64),
        }

    numpy_alg, numpy_state, numpy_pos, numpy_has_gauss, numpy_cached_gaussian = state["numpy"]
    python_version, python_internal_state, python_cached_gaussian = state["python"]
    torch_cuda_states = state.get("torch_cuda", [])
    torch_cuda_lengths = torch.tensor(
        [tensor.numel() for tensor in torch_cuda_states],
        dtype=torch.int64,
    )
    torch_cuda_flat = (
        torch.cat([tensor.cpu().to(torch.uint8).flatten() for tensor in torch_cuda_states])
        if torch_cuda_states
        else torch.zeros(0, dtype=torch.uint8)
    )

    return {
        "torch_cpu": state["torch_cpu"].cpu().to(torch.uint8),
        "torch_cuda_flat": torch_cuda_flat,
        "torch_cuda_lengths": torch_cuda_lengths,
        "numpy_alg": torch.tensor(list(numpy_alg.encode("utf-8")), dtype=torch.uint8),
        "numpy_state": torch.as_tensor(numpy_state.astype(np.int64), dtype=torch.int64),
        "numpy_pos": torch.tensor(int(numpy_pos), dtype=torch.int64),
        "numpy_has_gauss": torch.tensor(bool(numpy_has_gauss), dtype=torch.bool),
        "numpy_cached_gaussian": torch.tensor(float(numpy_cached_gaussian), dtype=torch.float64),
        "python_version": torch.tensor(int(python_version), dtype=torch.int64),
        "python_state": torch.tensor(list(python_internal_state), dtype=torch.int64),
        "python_has_gauss": torch.tensor(python_cached_gaussian is not None, dtype=torch.bool),
        "python_cached_gaussian": torch.tensor(
            float(python_cached_gaussian) if python_cached_gaussian is not None else 0.0,
            dtype=torch.float64,
        ),
    }


def _decode_rng_state(
    *,
    torch_cpu: torch.Tensor,
    torch_cuda_flat: torch.Tensor,
    torch_cuda_lengths: torch.Tensor,
    numpy_alg: torch.Tensor,
    numpy_state: torch.Tensor,
    numpy_pos: torch.Tensor,
    numpy_has_gauss: torch.Tensor,
    numpy_cached_gaussian: torch.Tensor,
    python_version: torch.Tensor,
    python_state: torch.Tensor,
    python_has_gauss: torch.Tensor,
    python_cached_gaussian: torch.Tensor,
) -> dict[str, Any]:
    numpy_alg_str = bytes(int(x) for x in numpy_alg.tolist()).decode("utf-8")
    numpy_tuple = (
        numpy_alg_str,
        np.asarray(numpy_state.tolist(), dtype=np.uint32),
        int(numpy_pos.item()),
        bool(numpy_has_gauss.item()),
        float(numpy_cached_gaussian.item()),
    )
    python_tuple = (
        int(python_version.item()),
        tuple(int(x) for x in python_state.tolist()),
        float(python_cached_gaussian.item()) if bool(python_has_gauss.item()) else None,
    )
    rng_state: dict[str, Any] = {
        "torch_cpu": torch_cpu.cpu().to(torch.uint8),
        "numpy": numpy_tuple,
        "python": python_tuple,
    }
    if torch_cuda_lengths.numel() > 0:
        torch_cuda_states = []
        offset = 0
        for length in torch_cuda_lengths.tolist():
            next_offset = offset + int(length)
            torch_cuda_states.append(torch_cuda_flat[offset:next_offset].clone())
            offset = next_offset
        rng_state["torch_cuda"] = torch_cuda_states
    return rng_state


class TrainAllPhasesState(torch.nn.Module):
    """Minimal persisted state for ``train_all_phases()``."""

    STATE_VERSION = 1
    SPECIAL_SERIALIZERS = {
        "mlflow_run_id": "_serialize_mlflow_run_id",
        "checkpoint_uuid": "_serialize_checkpoint_uuid",
    }
    SPECIAL_DESERIALIZERS = {
        "mlflow_run_id": "_deserialize_mlflow_run_id",
        "checkpoint_uuid": "_deserialize_checkpoint_uuid",
    }

    def __init__(
        self,
        *,
        job_idx: int = 0,
        next_epoch_anchor: int = 0,
        mlflow_run_id: str | None = None,
        checkpoint_uuid: str | None = None,
    ) -> None:
        super().__init__()
        self.job_idx = job_idx
        self.next_epoch_anchor = next_epoch_anchor
        self.mlflow_run_id = mlflow_run_id
        self.checkpoint_uuid = checkpoint_uuid
        self.register_buffer("_state_version", torch.tensor(self.STATE_VERSION, dtype=torch.int64))
        self.register_buffer("_job_idx", torch.tensor(job_idx, dtype=torch.int64))
        self.register_buffer(
            "_next_epoch_anchor",
            torch.tensor(next_epoch_anchor, dtype=torch.int64),
        )
        mlflow_run_id_bytes, mlflow_run_id_is_none = _encode_optional_string(mlflow_run_id)
        self.register_buffer("_mlflow_run_id_bytes", mlflow_run_id_bytes)
        self.register_buffer("_mlflow_run_id_is_none", mlflow_run_id_is_none)
        checkpoint_uuid_bytes, checkpoint_uuid_is_none = _encode_optional_string(checkpoint_uuid)
        self.register_buffer("_checkpoint_uuid_bytes", checkpoint_uuid_bytes)
        self.register_buffer("_checkpoint_uuid_is_none", checkpoint_uuid_is_none)

    def save(self, path: Path) -> None:
        self._state_version = torch.tensor(self.STATE_VERSION, dtype=torch.int64)
        self._job_idx = torch.tensor(int(self.job_idx), dtype=torch.int64)
        self._next_epoch_anchor = torch.tensor(int(self.next_epoch_anchor), dtype=torch.int64)
        self._run_special_serializers()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logging.info("Saved train_all_phases_state to %s", path)

    def load(self, path: Path) -> "TrainAllPhasesState":
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self._validate_state_version(state_dict)
        self._preload_variable_buffers_from_state_dict(state_dict)
        self.load_state_dict(state_dict, strict=True)
        self.job_idx = int(self._job_idx.item())
        self.next_epoch_anchor = int(self._next_epoch_anchor.item())
        self._run_special_deserializers()
        return self

    def _run_special_serializers(self) -> None:
        for field_name, serializer_name in self.SPECIAL_SERIALIZERS.items():
            serializer = _resolve_serializer(self, field_name, serializer_name)
            serializer()

    def _run_special_deserializers(self) -> None:
        for field_name, serializer_name in self.SPECIAL_DESERIALIZERS.items():
            serializer = _resolve_serializer(self, field_name, serializer_name)
            serializer()

    def _serialize_mlflow_run_id(self) -> None:
        self._mlflow_run_id_bytes, self._mlflow_run_id_is_none = _encode_optional_string(
            self.mlflow_run_id
        )

    def _serialize_checkpoint_uuid(self) -> None:
        self._checkpoint_uuid_bytes, self._checkpoint_uuid_is_none = _encode_optional_string(
            self.checkpoint_uuid
        )

    def _deserialize_mlflow_run_id(self) -> None:
        self.mlflow_run_id = _decode_optional_string(
            self._mlflow_run_id_bytes,
            self._mlflow_run_id_is_none,
        )

    def _deserialize_checkpoint_uuid(self) -> None:
        self.checkpoint_uuid = _decode_optional_string(
            self._checkpoint_uuid_bytes,
            self._checkpoint_uuid_is_none,
        )

    def _preload_variable_buffers_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._mlflow_run_id_bytes = state_dict["_mlflow_run_id_bytes"]
        self._checkpoint_uuid_bytes = state_dict["_checkpoint_uuid_bytes"]

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

    STATE_VERSION = 1
    SPECIAL_SERIALIZERS = {
        "rng_state": "_serialize_rng_state",
        "checkpoint_uuid": "_serialize_checkpoint_uuid",
    }
    SPECIAL_DESERIALIZERS = {
        "rng_state": "_deserialize_rng_state",
        "checkpoint_uuid": "_deserialize_checkpoint_uuid",
    }

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
        checkpoint_uuid: str | None = None,
    ) -> None:
        super().__init__()
        self.phase_epoch = phase_epoch
        self.nan_counter = nan_counter
        self.grad_norm_last_reduced_counter = grad_norm_last_reduced_counter
        self.stable_epochs = stable_epochs
        self.deterministic_mode_active = deterministic_mode_active
        self.seq_len_increase_in_batches = seq_len_increase_in_batches
        self.rng_state = rng_state or {}
        self.checkpoint_uuid = checkpoint_uuid
        self.register_buffer("_state_version", torch.tensor(self.STATE_VERSION, dtype=torch.int64))
        self.register_buffer("_phase_epoch", torch.tensor(phase_epoch, dtype=torch.int64))
        self.register_buffer("_nan_counter", torch.tensor(nan_counter, dtype=torch.int64))
        self.register_buffer(
            "_grad_norm_last_reduced_counter",
            torch.tensor(grad_norm_last_reduced_counter, dtype=torch.int64),
        )
        self.register_buffer("_stable_epochs", torch.tensor(stable_epochs, dtype=torch.int64))
        self.register_buffer(
            "_deterministic_mode_active",
            torch.tensor(deterministic_mode_active, dtype=torch.bool),
        )
        self.register_buffer(
            "_seq_len_increase_in_batches",
            torch.tensor(
                -1 if seq_len_increase_in_batches is None else seq_len_increase_in_batches,
                dtype=torch.int64,
            ),
        )
        self.register_buffer("_torch_cpu_rng", torch.zeros(0, dtype=torch.uint8))
        self.register_buffer("_torch_cuda_rng_flat", torch.zeros(0, dtype=torch.uint8))
        self.register_buffer("_torch_cuda_rng_lengths", torch.zeros(0, dtype=torch.int64))
        self.register_buffer("_numpy_alg", torch.zeros(0, dtype=torch.uint8))
        self.register_buffer("_numpy_state", torch.zeros(0, dtype=torch.int64))
        self.register_buffer("_numpy_pos", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_numpy_has_gauss", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_numpy_cached_gaussian", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_python_version", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_python_state", torch.zeros(0, dtype=torch.int64))
        self.register_buffer("_python_has_gauss", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_python_cached_gaussian", torch.tensor(0.0, dtype=torch.float64))
        checkpoint_uuid_bytes, checkpoint_uuid_is_none = _encode_optional_string(checkpoint_uuid)
        self.register_buffer("_checkpoint_uuid_bytes", checkpoint_uuid_bytes)
        self.register_buffer("_checkpoint_uuid_is_none", checkpoint_uuid_is_none)

    def save(self, path: Path) -> None:
        self._state_version = torch.tensor(self.STATE_VERSION, dtype=torch.int64)
        self._phase_epoch = torch.tensor(int(self.phase_epoch), dtype=torch.int64)
        self._nan_counter = torch.tensor(int(self.nan_counter), dtype=torch.int64)
        self._grad_norm_last_reduced_counter = torch.tensor(
            int(self.grad_norm_last_reduced_counter),
            dtype=torch.int64,
        )
        self._stable_epochs = torch.tensor(int(self.stable_epochs), dtype=torch.int64)
        self._deterministic_mode_active = torch.tensor(
            bool(self.deterministic_mode_active),
            dtype=torch.bool,
        )
        self._seq_len_increase_in_batches = torch.tensor(
            -1 if self.seq_len_increase_in_batches is None else int(self.seq_len_increase_in_batches),
            dtype=torch.int64,
        )
        self._run_special_serializers()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logging.info("Saved train_one_phase_state to %s", path)

    def load(self, path: Path) -> "TrainOnePhaseState":
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = self._filter_state_dict_for_registered_modules(state_dict)
        self._validate_state_version(state_dict)
        self._preload_variable_buffers_from_state_dict(state_dict)
        self.load_state_dict(state_dict, strict=True)
        self.phase_epoch = int(self._phase_epoch.item())
        self.nan_counter = int(self._nan_counter.item())
        self.grad_norm_last_reduced_counter = int(
            self._grad_norm_last_reduced_counter.item()
        )
        self.stable_epochs = int(self._stable_epochs.item())
        self.deterministic_mode_active = bool(self._deterministic_mode_active.item())
        seq_len_increase_in_batches = int(self._seq_len_increase_in_batches.item())
        self.seq_len_increase_in_batches = (
            None if seq_len_increase_in_batches < 0 else seq_len_increase_in_batches
        )
        self._run_special_deserializers()
        return self

    def _run_special_serializers(self) -> None:
        for field_name, serializer_name in self.SPECIAL_SERIALIZERS.items():
            serializer = _resolve_serializer(self, field_name, serializer_name)
            serializer()

    def _run_special_deserializers(self) -> None:
        for field_name, serializer_name in self.SPECIAL_DESERIALIZERS.items():
            serializer = _resolve_serializer(self, field_name, serializer_name)
            serializer()

    def _serialize_rng_state(self) -> None:
        encoded_rng_state = _encode_rng_state(self.rng_state)
        self._torch_cpu_rng = encoded_rng_state["torch_cpu"]
        self._torch_cuda_rng_flat = encoded_rng_state["torch_cuda_flat"]
        self._torch_cuda_rng_lengths = encoded_rng_state["torch_cuda_lengths"]
        self._numpy_alg = encoded_rng_state["numpy_alg"]
        self._numpy_state = encoded_rng_state["numpy_state"]
        self._numpy_pos = encoded_rng_state["numpy_pos"]
        self._numpy_has_gauss = encoded_rng_state["numpy_has_gauss"]
        self._numpy_cached_gaussian = encoded_rng_state["numpy_cached_gaussian"]
        self._python_version = encoded_rng_state["python_version"]
        self._python_state = encoded_rng_state["python_state"]
        self._python_has_gauss = encoded_rng_state["python_has_gauss"]
        self._python_cached_gaussian = encoded_rng_state["python_cached_gaussian"]

    def _serialize_checkpoint_uuid(self) -> None:
        self._checkpoint_uuid_bytes, self._checkpoint_uuid_is_none = _encode_optional_string(
            self.checkpoint_uuid
        )

    def _deserialize_rng_state(self) -> None:
        self.rng_state = _decode_rng_state(
            torch_cpu=self._torch_cpu_rng,
            torch_cuda_flat=self._torch_cuda_rng_flat,
            torch_cuda_lengths=self._torch_cuda_rng_lengths,
            numpy_alg=self._numpy_alg,
            numpy_state=self._numpy_state,
            numpy_pos=self._numpy_pos,
            numpy_has_gauss=self._numpy_has_gauss,
            numpy_cached_gaussian=self._numpy_cached_gaussian,
            python_version=self._python_version,
            python_state=self._python_state,
            python_has_gauss=self._python_has_gauss,
            python_cached_gaussian=self._python_cached_gaussian,
        )

    def _deserialize_checkpoint_uuid(self) -> None:
        self.checkpoint_uuid = _decode_optional_string(
            self._checkpoint_uuid_bytes,
            self._checkpoint_uuid_is_none,
        )

    def _preload_variable_buffers_from_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._torch_cpu_rng = state_dict["_torch_cpu_rng"]
        self._torch_cuda_rng_flat = state_dict["_torch_cuda_rng_flat"]
        self._torch_cuda_rng_lengths = state_dict["_torch_cuda_rng_lengths"]
        self._numpy_alg = state_dict["_numpy_alg"]
        self._numpy_state = state_dict["_numpy_state"]
        self._python_state = state_dict["_python_state"]
        self._checkpoint_uuid_bytes = state_dict["_checkpoint_uuid_bytes"]

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


def load_train_all_phases_state(path: Path) -> TrainAllPhasesState:
    return TrainAllPhasesState().load(path)


def load_train_all_phases_state_metadata(path: Path) -> dict[str, Any]:
    state = load_train_all_phases_state(path)
    return {
        "job_idx": state.job_idx,
        "next_epoch_anchor": state.next_epoch_anchor,
        "mlflow_run_id": state.mlflow_run_id,
        "checkpoint_uuid": state.checkpoint_uuid,
        "restart_state_path": str(path.resolve()),
        "hydra_output_dir": str(path.resolve().parent.resolve()),
    }


def load_train_one_phase_state(path: Path) -> TrainOnePhaseState:
    return TrainOnePhaseState().load(path)
