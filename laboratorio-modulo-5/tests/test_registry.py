from types import SimpleNamespace

import pytest

from laboratorio1.registry import (
    ensure_registered_version,
    find_registered_version,
    select_best_mlp_run,
    select_logged_model,
)


def make_run(
    run_id: str,
    *,
    model_type: str = "cost_sensitive_mlp",
    status: str = "FINISHED",
    f2: float = 0.7,
    pr_auc: float = 0.6,
    start_time: int = 1,
):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status=status, start_time=start_time),
        data=SimpleNamespace(
            tags={"model.type": model_type},
            metrics={"f2": f2, "pr_auc": pr_auc},
        ),
    )


def test_select_best_mlp_run_uses_f2_and_ignores_other_models():
    runs = [
        make_run("older", f2=0.75, start_time=1),
        make_run("winner", f2=0.8, start_time=2),
        make_run("auto", model_type="denoising_autoencoder", f2=0.99),
        make_run("failed", status="FAILED", f2=1.0),
    ]

    assert select_best_mlp_run(runs).info.run_id == "winner"


def test_select_best_mlp_run_requires_a_finished_candidate():
    with pytest.raises(RuntimeError, match="corrida MLP finalizada"):
        select_best_mlp_run([make_run("failed", status="FAILED")])


def test_select_logged_model_returns_latest_ready_model():
    models = [
        SimpleNamespace(
            source_run_id="winner",
            status="READY",
            creation_timestamp=1,
            model_uri="models:/old",
        ),
        SimpleNamespace(
            source_run_id="winner",
            status=SimpleNamespace(value="READY"),
            creation_timestamp=2,
            model_uri="models:/new",
        ),
        SimpleNamespace(
            source_run_id="other",
            status="READY",
            creation_timestamp=3,
            model_uri="models:/other",
        ),
    ]

    assert select_logged_model(models, "winner").model_uri == "models:/new"


def test_find_registered_version_matches_model_hash():
    versions = [
        SimpleNamespace(version="1", tags={"model.sha256": "first"}),
        SimpleNamespace(version="2", tags={"model.sha256": "second"}),
    ]

    assert find_registered_version(versions, "second").version == "2"
    assert find_registered_version(versions, "missing") is None


def test_ensure_registered_version_reuses_existing(monkeypatch):
    existing = SimpleNamespace(version="3", tags={"model.sha256": "same"})
    client = SimpleNamespace(search_model_versions=lambda _: [existing])

    def unexpected_registration(*args, **kwargs):
        raise AssertionError("No debe crear otra versión para el mismo modelo")

    monkeypatch.setattr("laboratorio1.registry.mlflow.register_model", unexpected_registration)

    version, created = ensure_registered_version(
        client,
        model_uri="models:/logged",
        source_run_id="run-1",
        model_sha256="same",
        registered_model_name="deteccion_fraude_mlp",
    )

    assert version is existing
    assert created is False


def test_ensure_registered_version_creates_a_new_version(monkeypatch):
    captured = {}
    created_version = SimpleNamespace(version="1", tags={})
    client = SimpleNamespace(search_model_versions=lambda _: [])

    def fake_register(model_uri, name, **kwargs):
        captured.update(model_uri=model_uri, name=name, **kwargs)
        return created_version

    monkeypatch.setattr("laboratorio1.registry.mlflow.register_model", fake_register)

    version, created = ensure_registered_version(
        client,
        model_uri="models:/logged",
        source_run_id="run-1",
        model_sha256="new-hash",
        registered_model_name="deteccion_fraude_mlp",
    )

    assert version is created_version
    assert created is True
    assert captured["model_uri"] == "models:/logged"
    assert captured["name"] == "deteccion_fraude_mlp"
    assert captured["tags"] == {
        "model.sha256": "new-hash",
        "model.role": "winner",
        "selection.metric": "f2",
        "source.run_id": "run-1",
    }
