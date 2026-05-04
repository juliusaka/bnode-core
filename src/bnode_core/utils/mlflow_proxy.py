"""MLflow proxy utilities with per-run parameter caching."""

import mlflow


class mlflow_proxy:
    """Proxy MLflow access and cache params for the active run."""

    _cached_run_id: str | None = None
    _cached_params: dict[str, str] = {}

    @classmethod
    def _get_active_run_id(cls) -> str | None:
        active_run = mlflow.active_run()
        if active_run is None:
            return None
        return active_run.info.run_id

    @classmethod
    def _get_active_run_params(cls) -> dict[str, str]:
        active_run_id = cls._get_active_run_id()
        if active_run_id is None:
            cls.reset_cache()
            return cls._cached_params
        if cls._cached_run_id != active_run_id:
            cls._cached_run_id = active_run_id
            cls._cached_params = dict(mlflow.get_run(active_run_id).data.params)
        return cls._cached_params

    @classmethod
    def log_param(cls, key: str, value) -> None:
        """Log an MLflow parameter only if absent; raise on conflicts."""
        params = cls._get_active_run_params()
        value_str = str(value)
        if key in params:
            if params[key] != value_str:
                raise ValueError(
                    f"Cannot change existing MLflow param '{key}' from {params[key]!r} to {value_str!r}"
                )
            return
        mlflow.log_param(key, value)
        params[key] = value_str

    @classmethod
    def log_metric(cls, key: str, value, *, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    @classmethod
    def log_metrics(cls, metrics: dict[str, float], *, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    @classmethod
    def set_tag_if_active(cls, key: str, value) -> None:
        """Set an MLflow tag if there is an active run."""
        if cls._get_active_run_id() is not None:
            mlflow.set_tag(key, value)

    @classmethod
    def reset_cache(cls) -> None:
        cls._cached_run_id = None
        cls._cached_params = {}
