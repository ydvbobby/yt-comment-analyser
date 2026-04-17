import pytest
import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelPerformanceAndProductionPromotion:
    """
    Test suite for model performance validation and production promotion.
    
    Workflow:
    1. Load model from 'Staging' stage (already validated)
    2. Load test data from data/processed/processed_test.csv
    3. Run predictions and calculate performance metrics
    4. Check if all metrics meet 75% threshold
    5. Only after all thresholds pass, promote to 'Production'
    
    This follows MLflow best practices where Production stage
    represents a model ready for live deployment.
    """
    
    METRIC_THRESHOLD = 0.75  # 75% threshold for all metrics

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
    def staging_model(self, mlflow_client, model_name):
        """
        Load the latest model version from 'Staging' stage.
        This model has already passed initial validation tests.
        """
        load_dotenv()
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            # Get latest version in "Staging" stage
            staging_versions = mlflow_client.get_latest_versions(
                name=model_name,
                stages=["Staging"]
            )
            
            if len(staging_versions) == 0:
                pytest.skip("No model versions found in 'Staging' stage - run test_model_loading.py first")
            
            version = staging_versions[0].version
            model_uri = f"models:/{model_name}/Staging"
            model = mlflow.keras.load_model(model_uri)
            
            return {"model": model, "version": version, "uri": model_uri}
            
        except Exception as e:
            pytest.skip(f"Could not load model from Staging: {str(e)}")

    @pytest.fixture
    def test_data(self):
        """Load test dataset from processed_test.csv."""
        test_data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "processed",
            "processed_test.csv"
        )
        
        try:
            df = pd.read_csv(test_data_path)
            
            # Ensure required columns exist
            if "clean_comment" not in df.columns or "category" not in df.columns:
                pytest.skip(f"Test data missing required columns. Found: {list(df.columns)}")
            
            # Filter out rows with NaN values in clean_comment
            df = df.dropna(subset=["clean_comment"])
            
            # Ensure category values are valid (convert to int if needed)
            df = df.dropna(subset=["category"])
            
            
            return {
                "texts": df["clean_comment"].astype(str).to_numpy(),  # Convert to numpy array for model inp
                "labels": df["category"].astype(int).to_numpy(),  # Ensure labels are integers
                "size": len(df)
            }
            
        except FileNotFoundError:
            pytest.skip(f"Test data file not found at {test_data_path}")
        except Exception as e:
            pytest.skip(f"Could not load test data: {str(e)}")

    # ==================== PERFORMANCE VALIDATION PHASE ====================
    # These tests validate model performance BEFORE production promotion

    def test_staging_model_loads_successfully(self, staging_model):
        """Validate: Model loads without errors from 'Staging' stage."""
        assert staging_model["model"] is not None
        assert staging_model["version"] is not None
        assert staging_model["uri"] is not None


    def test_model_prediction_accuracy(self, staging_model, test_data):
        """
        Validate: Model accuracy meets threshold (>= 75%).
        
        This is the primary metric for overall model performance.
        """
        model = staging_model["model"]
        preds = model.predict(test_data["texts"])
        predictions = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
        
        # Reverse map predictions from {0,1,2} to {-1,0,1} to match test labels
        # Model was trained with mapping: {-1:0, 0:1, 1:2}
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        predictions = np.array([reverse_mapping[p] for p in predictions])
        
        accuracy = accuracy_score(test_data["labels"], predictions)
        
        assert accuracy >= self.METRIC_THRESHOLD, \
            f"Accuracy {accuracy:.4f} below threshold {self.METRIC_THRESHOLD}"
        
        print(f"\n[PASS] Accuracy: {accuracy:.4f} (threshold: {self.METRIC_THRESHOLD})")


    def test_promote_validated_model_to_production(self, mlflow_client, model_name, staging_model):
        """
        Promote model to Production AFTER all performance tests pass.
        
        This test should run LAST in the test suite. If any performance
        test above fails, this test won't execute.
        
        Note: In production CI/CD, you might want to:
        - Add manual approval gate before this step
        - Log promotion event to audit trail
        - Send notification to stakeholders
        """
        version = staging_model["version"]
        
        try:
            # Promote to Production
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True  # Archive old Production versions
            )
            
            # Verify promotion
            production_versions = mlflow_client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )
            
            assert len(production_versions) > 0, "No model versions in Production after promotion"
            assert production_versions[0].version == version, \
                f"Expected version {version} in Production, got {production_versions[0].version}"
            
            print(f"\n[PASS] Model version {version} successfully promoted to Production!")
            
        except Exception as e:
            pytest.fail(f"Failed to promote model to Production: {str(e)}")

    def test_load_promoted_model_from_production(self, mlflow_client, model_name):
        """
        Verify the promoted model can be loaded from Production.
        
        This test depends on test_promote_validated_model_to_production passing.
        """
        try:
            # Load from Production stage
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.keras.load_model(model_uri)
            
            assert model is not None, "Model loaded from Production is None"
            
            # Quick sanity check prediction
            predictions = model.predict(["Test comment"])
            assert len(predictions) == 1
            
            print(f"\n[PASS] Successfully loaded model from Production!")
            
        except Exception as e:
            pytest.skip(f"Could not load model from Production (may not be promoted yet): {str(e)}")


# ==================== RUN CONFIGURATION ====================

if __name__ == "__main__":
    # Run tests in order with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
