import pytest
import requests
import time

BASE_URL = "http://127.0.0.1:8000"


def wait_for_server():
    """Wait until FastAPI server is ready."""
    for _ in range(10):
        try:
            r = requests.get(f"{BASE_URL}/docs")
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    pytest.fail("Server did not start in time")


class TestBackendAPI:
    
    @classmethod
    def setup_class(cls):
        wait_for_server()

    def test_predict_endpoint(self):
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": ["This is a great product!"]},
            timeout=10
        )

        assert response.status_code == 200

        data = response.json()

        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
        assert data["predictions"][0] in [-1, 0, 1]

    def test_pie_chart_endpoint(self):
        response = requests.post(
            f"{BASE_URL}/pie-chart",
            json={"positive": 50, "neutral": 30, "negative": 20},
            timeout=10
        )

        assert response.status_code == 200
        assert "image/png" in response.headers.get("content-type", "")
        assert len(response.content) > 0