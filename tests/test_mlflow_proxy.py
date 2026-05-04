from types import SimpleNamespace

import pytest

from bnode_core.utils.mlflow_proxy import mlflow_proxy


def _fake_run(run_id: str, params: dict[str, str]):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(params=params),
    )


def test_mlflow_proxy_log_param_caches_active_run(monkeypatch):
    logged_params: list[tuple[str, object]] = []
    get_run_calls: list[str] = []
    active_run = _fake_run("run-1", {"existing": "value"})

    monkeypatch.setattr("bnode_core.utils.mlflow_proxy.mlflow.active_run", lambda: active_run)

    def fake_get_run(run_id: str):
        get_run_calls.append(run_id)
        return _fake_run(run_id, {"existing": "value"})

    monkeypatch.setattr("bnode_core.utils.mlflow_proxy.mlflow.get_run", fake_get_run)
    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.log_param",
        lambda key, value: logged_params.append((key, value)),
    )
    mlflow_proxy.reset_cache()

    mlflow_proxy.log_param("new_param", 1)
    mlflow_proxy.log_param("new_param", 1)
    mlflow_proxy.log_param("other_param", "x")

    assert get_run_calls == ["run-1"]
    assert logged_params == [("new_param", 1), ("other_param", "x")]


def test_mlflow_proxy_log_param_rejects_conflicts(monkeypatch):
    active_run = _fake_run("run-1", {"existing": "value"})

    monkeypatch.setattr("bnode_core.utils.mlflow_proxy.mlflow.active_run", lambda: active_run)
    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.get_run",
        lambda run_id: _fake_run(run_id, {"existing": "value"}),
    )
    mlflow_proxy.reset_cache()

    with pytest.raises(ValueError, match="Cannot change existing MLflow param"):
        mlflow_proxy.log_param("existing", "different")


def test_mlflow_proxy_refreshes_cache_when_run_changes(monkeypatch):
    active_runs = [
        _fake_run("run-1", {"param": "one"}),
        _fake_run("run-2", {"param": "two"}),
    ]
    current_run = {"value": active_runs[0]}
    get_run_calls: list[str] = []

    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.active_run",
        lambda: current_run["value"],
    )

    def fake_get_run(run_id: str):
        get_run_calls.append(run_id)
        if run_id == "run-1":
            return _fake_run(run_id, {"param": "one"})
        return _fake_run(run_id, {"param": "two"})

    monkeypatch.setattr("bnode_core.utils.mlflow_proxy.mlflow.get_run", fake_get_run)
    mlflow_proxy.reset_cache()

    mlflow_proxy.log_param("param", "one")
    current_run["value"] = active_runs[1]
    mlflow_proxy.log_param("param", "two")

    assert get_run_calls == ["run-1", "run-2"]


def test_mlflow_proxy_set_tag_if_active(monkeypatch):
    set_tags: list[tuple[str, object]] = []
    active_run = {"value": None}

    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.active_run",
        lambda: active_run["value"],
    )
    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.set_tag",
        lambda key, value: set_tags.append((key, value)),
    )
    mlflow_proxy.reset_cache()

    mlflow_proxy.set_tag_if_active("key", "ignored")
    active_run["value"] = _fake_run("run-1", {})
    mlflow_proxy.set_tag_if_active("key", "value")

    assert set_tags == [("key", "value")]


def test_mlflow_proxy_log_metric_delegates(monkeypatch):
    logged_metrics: list[tuple[str, float, int | None]] = []

    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.log_metric",
        lambda key, value, step=None: logged_metrics.append((key, value, step)),
    )

    mlflow_proxy.log_metric("loss", 1.25, step=7)

    assert logged_metrics == [("loss", 1.25, 7)]


def test_mlflow_proxy_log_metrics_delegates(monkeypatch):
    logged_metric_batches: list[tuple[dict[str, float], int | None]] = []

    monkeypatch.setattr(
        "bnode_core.utils.mlflow_proxy.mlflow.log_metrics",
        lambda metrics, step=None: logged_metric_batches.append((metrics, step)),
    )

    metrics = {"loss": 1.25, "rmse": 0.5}
    mlflow_proxy.log_metrics(metrics, step=3)

    assert logged_metric_batches == [(metrics, 3)]
