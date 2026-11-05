from fastapi.testclient import TestClient
from api.app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_categories():
    with TestClient(app) as client:
        response = client.get("/categories")
        assert response.status_code == 200
        assert set(response.json()["categories"]) == {
            "road", "waste", "water", "safety"
        }


def test_predict_shape():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "There is a pothole on my street"}
        )

        assert response.status_code == 200

        body = response.json()

        assert set(body.keys()) == {"predicted_category", "confidence"}
        assert body["predicted_category"] in {
            "road", "waste", "water", "safety"
        }
        assert 0.0 <= body["confidence"] <= 100.0


def test_predict_rejects_missing_text():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422