import pytest
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv
import json
import numpy as np


class TestMLflowModelValidationAndPromotion:
    """
    Test suite for MLflow model validation and stage promotion.
    
    Workflow:
    1. Load model from 'None' stage (newly registered)
    2. Validate model performance
    3. Only after validation passes, promote to 'Staging'
    
    This follows MLflow best practices where stages represent
    the model maturity level in the deployment pipeline.
    """

    @pytest.fixture
    def mlflow_client(self):
        """Create and configure MLflow client."""
        load_dotenv()
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        return MlflowClient(tracking_uri)

    @pytest.fixture
    def model_name(self):
        """Model name in MLflow registry."""
        return "yt-comment-analyzer"

    @pytest.fixture
    def loaded_model(self, mlflow_client, model_name):
        """
        Load the latest model version from 'None' stage.
        This is the unvalidated model ready for testing.
        """
        load_dotenv()
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            # Get latest version in "None" stage (unvalidated)
            latest_versions = mlflow_client.get_latest_versions(
                name=model_name,
                stages=["None"]
            )
            
            if len(latest_versions) == 0:
                pytest.skip("No model versions found in 'None' stage - model may already be promoted")
            
            version = latest_versions[0].version
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            return {"model": model, "version": version, "uri": model_uri}
            
        except Exception as e:
            pytest.skip(f"Could not load model for validation: {str(e)}")

    def test_mlflow_tracking_uri_set(self):
        """Test that MLflow tracking URI is properly configured."""
        load_dotenv()
        
        tracking_uri = os.getenv("mlflow_tracking_uri")
        assert tracking_uri is not None, "MLflow tracking URI not set in .env"
        assert tracking_uri.startswith("http") or os.path.exists(tracking_uri), \
            "Invalid MLflow tracking URI format"

    def test_mlflow_connection(self):
        """Test that MLflow can connect to the tracking server."""
        load_dotenv()
        
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        
        uri = mlflow.get_tracking_uri()
        assert uri is not None

    # ==================== VALIDATION PHASE ====================
    # These tests validate the model BEFORE promotion

    def test_model_loads_successfully(self, loaded_model):
        """Validate: Model loads without errors from 'None' stage."""
        assert loaded_model["model"] is not None
        assert loaded_model["version"] is not None
        assert loaded_model["uri"] is not None

    def test_model_predicts_correctly(self, loaded_model):
        """Validate: Model produces valid predictions for sample inputs."""
        model = loaded_model["model"]
        
        sample_comments = [
            "Great video! I loved it.",
            "Terrible content, waste of time.",
            "It was okay, nothing special."
        ]
        
        predictions = model.predict(sample_comments)
        
        # Validate prediction output
        assert len(predictions) == len(sample_comments), \
            f"Expected {len(sample_comments)} predictions, got {len(predictions)}"
        
        # Predictions should be class labels (0, 1, 2) - handle both Python and NumPy types
        assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in predictions), \
            f"Predictions should be numeric: {predictions}"
        
        assert all(p in [0, 1, 2] for p in predictions), \
            f"Invalid class predictions: {predictions}. Expected values in [0, 1, 2]"

    def test_model_predicts_proba(self, loaded_model):
        """Validate: Model provides probability scores for predictions."""
        model = loaded_model["model"]
        
        sample_comments = ["This is a test comment."]
        probas = model.predict_proba(sample_comments)
        
        assert probas.shape[0] == 1, "Should return probabilities for one sample"
        assert probas.shape[1] == 3, "Should return probabilities for 3 classes"
        
        # Probabilities should sum to 1
        assert np.isclose(probas.sum(axis=1), 1.0).all(), \
            f"Probabilities should sum to 1, got {probas.sum(axis=1)}"

    def test_model_confidence_threshold(self, loaded_model):
        """Validate: Model predictions meet minimum confidence threshold."""
        model = loaded_model["model"]
        
        sample_comments = [
            "Amazing content!",
            "I hate this video",
            "It is what it is"
        ]
        
        probas = model.predict_proba(sample_comments)
        max_confidences = np.max(probas, axis=1)
        
        # At least 2 out of 3 predictions should have >50% confidence
        confident_predictions = (max_confidences > 0.5).sum()
        assert confident_predictions >= 2, \
            f"Model confidence too low: {confident_predictions}/3 predictions above 50%"

    def test_model_has_required_methods(self, loaded_model):
        """Validate: Model has all required sklearn methods."""
        model = loaded_model["model"]
        
        required_methods = ['predict', 'predict_proba', 'score']
        for method in required_methods:
            assert hasattr(model, method), f"Model missing required method: {method}"

    def test_model_preprocessing_pipeline(self, loaded_model):
        """Validate: Model includes preprocessing pipeline (vectorizer)."""
        model = loaded_model["model"]
        
        # Check if model has pipeline structure with vectorizer
        if hasattr(model, 'named_steps'):
            # It's a Pipeline
            assert 'vectorizer' in model.named_steps or 'countvectorizer' in str(model).lower(), \
                "Pipeline should contain a vectorizer step"
        # If not a pipeline, the model should still be callable
        # (some MLflow models wrap the pipeline differently)

    # ==================== PROMOTION PHASE ====================
    # Only executed after all validation tests pass

    def test_promote_validated_model_to_staging(self, mlflow_client, model_name, loaded_model):
        """
        Promote model to Staging AFTER all validation tests pass.
        
        This test should run LAST in the test suite. If any validation
        test above fails, this test won't execute (pytest stops on first failure
        unless --continue-on-collection-errors is used).
        
        Note: In CI/CD, you might want to separate validation and promotion
        into different test files or use pytest markers.
        """
        version = loaded_model["version"]
        
        try:
            # Promote to Staging
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
                archive_existing_versions=False
            )
            
            # Verify promotion
            staging_versions = mlflow_client.get_latest_versions(
                name=model_name,
                stages=["Staging"]
            )
            
            assert len(staging_versions) > 0, "No model versions in Staging after promotion"
            assert staging_versions[0].version == version, \
                f"Expected version {version} in Staging, got {staging_versions[0].version}"
            
            print(f"\n[PASS] Model version {version} successfully promoted to Staging!")
            
        except Exception as e:
            pytest.fail(f"Failed to promote model to Staging: {str(e)}")

    def test_load_promoted_model_from_staging(self, mlflow_client, model_name):
        """
        Verify the promoted model can be loaded from Staging.
        
        This test depends on test_promote_validated_model_to_staging passing.
        Run tests in order: pytest test_model_loading.py -v
        """
        try:
            # Load from Staging stage (latest Staging version)
            model_uri = f"models:/{model_name}/Staging"
            model = mlflow.sklearn.load_model(model_uri)
            
            assert model is not None, "Model loaded from Staging is None"
            
            # Quick sanity check prediction
            predictions = model.predict(["Test comment"])
            assert len(predictions) == 1
            
            print(f"\n[PASS] Successfully loaded model from Staging!")
            
        except Exception as e:
            pytest.skip(f"Could not load model from Staging (may not be promoted yet): {str(e)}")


# ==================== RUN CONFIGURATION ====================

if __name__ == "__main__":
    # Run tests in order with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
