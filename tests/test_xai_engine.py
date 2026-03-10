"""
tests/test_xai_engine.py
Unit and integration tests for XAI Engine components.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def breast_cancer_dataset():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, list(data.feature_names), list(data.target_names)


@pytest.fixture(scope="module")
def trained_mlp(breast_cancer_dataset):
    from deep_model import MLPClassifier
    X_train, _, y_train, _, _, _ = breast_cancer_dataset
    model = MLPClassifier(epochs=20, hidden_dims=[32, 16], verbose=False)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def xai_engine(trained_mlp, breast_cancer_dataset):
    from xai_engine import XAIEngine
    X_train, _, _, _, feature_names, class_names = breast_cancer_dataset
    engine = XAIEngine(
        model=trained_mlp,
        X_train=X_train,
        feature_names=feature_names,
        class_names=class_names,
        shap_type="kernel",
    )
    return engine


# ─── Model Tests ───────────────────────────────────────────────────────────

class TestMLPClassifier:
    def test_fit_predict(self, breast_cancer_dataset):
        from deep_model import MLPClassifier
        X_train, X_test, y_train, y_test, _, _ = breast_cancer_dataset
        model = MLPClassifier(epochs=20, verbose=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_proba_shape(self, trained_mlp, breast_cancer_dataset):
        _, X_test, _, _, _, class_names = breast_cancer_dataset
        proba = trained_mlp.predict_proba(X_test)
        assert proba.shape == (len(X_test), len(class_names))

    def test_proba_sums_to_one(self, trained_mlp, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        proba = trained_mlp.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_score_reasonable(self, trained_mlp, breast_cancer_dataset):
        _, X_test, _, y_test, _, _ = breast_cancer_dataset
        acc = trained_mlp.score(X_test, y_test)
        assert acc > 0.6, f"Expected accuracy > 60%, got {acc:.2%}"

    def test_classes_set(self, trained_mlp):
        assert trained_mlp.classes_ is not None
        assert len(trained_mlp.classes_) == 2


class TestOtherModels:
    @pytest.mark.parametrize("model_name", ["cnn", "lstm", "transformer"])
    def test_model_factory(self, model_name, breast_cancer_dataset):
        from deep_model import get_model
        X_train, X_test, y_train, _, _, _ = breast_cancer_dataset
        model = get_model(model_name, epochs=10, verbose=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_autoencoder(self, breast_cancer_dataset):
        from deep_model import AutoEncoder
        X_train, X_test, _, _, _, _ = breast_cancer_dataset
        ae = AutoEncoder(latent_dim=8, epochs=10)
        ae.fit(X_train)
        scores = ae.anomaly_score(X_test)
        preds  = ae.predict(X_test)
        assert len(scores) == len(X_test)
        assert set(preds).issubset({0, 1})


# ─── SHAP Tests ────────────────────────────────────────────────────────────

class TestSHAPExplainer:
    def test_shap_values_shape(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        exp = xai_engine.shap_explainer.explain(X_test[:3])
        sv  = exp["shap_values"]
        # For binary classification, sv can be a list of 2 arrays or single array
        if isinstance(sv, list):
            assert sv[0].shape[1] == X_test.shape[1]
        else:
            assert sv.shape[1] == X_test.shape[1]

    def test_feature_importance_sorted(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        exp = xai_engine.shap_explainer.explain(X_test[:5])
        df  = xai_engine.shap_explainer.get_feature_importance(exp)
        assert df["importance"].is_monotonic_decreasing

    def test_feature_importance_non_negative(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        exp = xai_engine.shap_explainer.explain(X_test[:5])
        df  = xai_engine.shap_explainer.get_feature_importance(exp)
        assert (df["importance"] >= 0).all()


# ─── LIME Tests ────────────────────────────────────────────────────────────

class TestLIMEExplainer:
    def test_lime_explanation_produced(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        exp = xai_engine.lime_explainer.explain(X_test[0], num_samples=200)
        assert "lime_exp" in exp

    def test_lime_feature_importance_columns(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        exp = xai_engine.lime_explainer.explain(X_test[0], num_samples=200)
        df  = xai_engine.lime_explainer.get_feature_importance(exp)
        assert "feature" in df.columns
        assert "importance" in df.columns


# ─── XAI Engine Integration Tests ─────────────────────────────────────────

class TestXAIEngine:
    def test_explain_returns_required_keys(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        report = xai_engine.explain(X_test[0])
        required_keys = [
            "prediction", "confidence", "shap_explanation", "lime_explanation",
            "shap_importance", "lime_importance", "narrative",
            "shap_figure", "lime_figure",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_confidence_in_range(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        report = xai_engine.explain(X_test[0])
        assert 0.0 <= report["confidence"] <= 1.0

    def test_prediction_is_valid_class(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, class_names = breast_cancer_dataset
        report = xai_engine.explain(X_test[0])
        assert report["prediction"] in class_names

    def test_narrative_is_string(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        report = xai_engine.explain(X_test[0])
        assert isinstance(report["narrative"], str)
        assert len(report["narrative"]) > 50

    def test_batch_explain_count(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        results = xai_engine.batch_explain(X_test, n_samples=3)
        assert len(results) == 3

    def test_generate_report_creates_file(self, xai_engine, breast_cancer_dataset, tmp_path):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        report = xai_engine.explain(X_test[0])
        output_path = str(tmp_path / "test_report.html")
        result = xai_engine.generate_report(report, save_path=output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            content = f.read()
        assert "Explainable AI Report" in content
        assert "SHAP" in content
        assert "LIME" in content

    def test_global_explanation_keys(self, xai_engine, breast_cancer_dataset):
        _, X_test, _, _, _, _ = breast_cancer_dataset
        global_exp = xai_engine.global_explanation(X_test[:20])
        assert "feature_importance" in global_exp
        assert "global_summary_figure" in global_exp
