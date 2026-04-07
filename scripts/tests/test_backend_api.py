"""Simple integration tests for the FastAPI backend API endpoints.

These tests assume the FastAPI server is running at http://127.0.0.1:8000
Run the server first: uvicorn backend.app:app --host 127.0.0.1 --port 8000
Then run tests: pytest scripts/tests/test_backend_api.py
"""

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


class TestBackendAPI:
    """Integration tests for the FastAPI backend endpoints."""
    
    def test_predict_endpoint(self):
        """Test the /predict endpoint returns sentiment predictions."""
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": ["This is a great product!"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        # Verify prediction is in original label space {-1, 0, 1}
        assert data["predictions"][0] in [-1, 0, 1]
    
    def test_pie_chart_endpoint(self):
        """Test the /pie-chart endpoint generates a PNG image."""
        response = requests.post(
            f"{BASE_URL}/pie-chart",
            json={"positive": 50, "neutral": 30, "negative": 20}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 0
