"""MLflow helper utilities for trainer logging."""

import mlflow


def _get_active_run_params() -> dict[str, str]:
    active_run = mlflow.active_run()
    if active_run is None:
        return {}
    return dict(mlflow.get_run(active_run.info.run_id).data.params)


def _log_mlflow_param(
    key: str,
    value,
    *,
    existing_params: dict[str, str] | None = None,
) -> dict[str, str]:
    """Log an MLflow parameter only if absent; raise if it conflicts with an existing value."""
    params = existing_params if existing_params is not None else _get_active_run_params()
    value_str = str(value)
    if key in params:
        if params[key] != value_str:
            raise ValueError(
                f"Cannot change existing MLflow param '{key}' from {params[key]!r} to {value_str!r}"
            )
        return params
    mlflow.log_param(key, value)
    params[key] = value_str
    return params


def _set_mlflow_tag_if_active(key: str, value) -> None:
    """Set an MLflow tag if there is an active run."""
    if mlflow.active_run() is not None:
        mlflow.set_tag(key, value)
