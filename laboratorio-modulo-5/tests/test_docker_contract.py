from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_serves_only_the_mlp_on_port_8080():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "EXPOSE 8080" in dockerfile
    assert '"mlflow", "models", "serve"' in dockerfile
    assert '"/opt/mlflow/model"' in dockerfile
    assert "app:app" not in dockerfile
    assert "EXPOSE 8000" not in dockerfile


def test_compose_publishes_only_the_model_port():
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert '"8080:8080"' in compose
    assert '"8000:8000"' not in compose
