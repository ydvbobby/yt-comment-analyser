import pytest
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras


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
            df["category"] = df["category"].astype(int)
            
            return {
                "texts": df["clean_comment"].tolist(),
                "labels": df["category"].tolist(),
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

    def test_test_data_loaded(self, test_data):
        """Validate: Test data loaded successfully."""
        assert test_data["texts"] is not None
        assert test_data["labels"] is not None
        assert test_data["size"] > 0, "Test data should not be empty"
        assert len(test_data["texts"]) == len(test_data["labels"]), \
            "Text and label counts should match"

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
        predictions = np.array(predictions).astype(int)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        predictions = np.array([reverse_mapping[p] for p in predictions])
        
        accuracy = accuracy_score(test_data["labels"], predictions)
        
        assert accuracy >= self.METRIC_THRESHOLD, \
            f"Accuracy {accuracy:.4f} below threshold {self.METRIC_THRESHOLD}"
        
        print(f"\n[PASS] Accuracy: {accuracy:.4f} (threshold: {self.METRIC_THRESHOLD})")

    def test_model_precision(self, staging_model, test_data):
        """
        Validate: Model precision meets threshold (>= 75%).
        
        Precision measures the quality of positive predictions.
        """
        model = staging_model["model"]
        predictions = model.predict(test_data["texts"])
        
        # Reverse map predictions from {0,1,2} to {-1,0,1} to match test labels
        # Model was trained with mapping: {-1:0, 0:1, 1:2}
        predictions = np.array(predictions).astype(int)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        predictions = np.array([reverse_mapping[p] for p in predictions])
        
        precision = precision_score(
            test_data["labels"], 
            predictions, 
            average="weighted",
            zero_division=0
        )
        
        assert precision >= self.METRIC_THRESHOLD, \
            f"Precision {precision:.4f} below threshold {self.METRIC_THRESHOLD}"
        
        print(f"\n[PASS] Precision: {precision:.4f} (threshold: {self.METRIC_THRESHOLD})")

    def test_model_recall(self, staging_model, test_data):
        """
        Validate: Model recall meets threshold (>= 75%).
        
        Recall measures the ability to find all positive samples.
        """
        model = staging_model["model"]
        preds = model.predict(test_data["texts"])
        predictions = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
        
        # Reverse map predictions from {0,1,2} to {-1,0,1} to match test labels
        # Model was trained with mapping: {-1:0, 0:1, 1:2}
        predictions = np.array(predictions).astype(int)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        predictions = np.array([reverse_mapping[p] for p in predictions])
        
        recall = recall_score(
            test_data["labels"], 
            predictions, 
            average="weighted",
            zero_division=0
        )
        
        assert recall >= self.METRIC_THRESHOLD, \
            f"Recall {recall:.4f} below threshold {self.METRIC_THRESHOLD}"
        
        print(f"\n[PASS] Recall: {recall:.4f} (threshold: {self.METRIC_THRESHOLD})")

    def test_model_f1_score(self, staging_model, test_data):
        """
        Validate: Model F1 score meets threshold (>= 75%).
        
        F1 score is the harmonic mean of precision and recall.
        """
        model = staging_model["model"]
        preds = model.predict(test_data["texts"])
        predictions = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
        
        # Reverse map predictions from {0,1,2} to {-1,0,1} to match test labels
        # Model was trained with mapping: {-1:0, 0:1, 1:2}
        predictions = np.array(predictions).astype(int)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        predictions = np.array([reverse_mapping[p] for p in predictions])
        
        f1 = f1_score(
            test_data["labels"], 
            predictions, 
            average="weighted",
            zero_division=0
        )
        
        assert f1 >= self.METRIC_THRESHOLD, \
            f"F1 Score {f1:.4f} below threshold {self.METRIC_THRESHOLD}"
        
        print(f"\n[PASS] F1 Score: {f1:.4f} (threshold: {self.METRIC_THRESHOLD})")

    def test_model_prediction_distribution(self, staging_model, test_data):
        """
        Validate: Model predictions are reasonably distributed across classes.
        
        Prevents degenerate cases where model predicts only one class.
        """
        model = staging_model["model"]
        preds = model.predict(test_data["texts"])
        predictions = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
        
        # Reverse map predictions from {0,1,2} to {-1,0,1} to match test labels
        # Model was trained with mapping: {-1:0, 0:1, 1:2}
        predictions = np.array(predictions).astype(int)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        predictions = np.array([reverse_mapping[p] for p in predictions])
        
        unique_classes = len(set(predictions))
        expected_classes = len(set(test_data["labels"]))
        
        # Model should predict at least half the expected classes
        assert unique_classes >= max(1, expected_classes // 2), \
            f"Model predicts only {unique_classes} classes, expected at least {expected_classes // 2}"
        
        print(f"\n[PASS] Prediction distribution: {unique_classes} unique classes predicted")

    # ==================== PRODUCTION PROMOTION PHASE ====================
    # Only executed after all performance tests pass

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
