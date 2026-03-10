"""
healthcare_xai.py - XAI for Healthcare Applications
=====================================================
Demonstrates Explainable AI for clinical decision support.

Use Cases:
1. Disease diagnosis explainability (e.g., diabetes, heart disease)
2. Patient risk stratification with transparent reasoning
3. Drug interaction prediction with feature attribution

Compliance Context:
- EU AI Act (High-Risk AI) - requires human oversight and explainability
- FDA guidance on AI/ML-based Software as a Medical Device (SaMD)
- HIPAA-compliant audit trails via XAI narratives

Author: Pranay M Mahendrakar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_engine import XAIEngine
from deep_model import MLPClassifier, get_model


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_breast_cancer_data():
    """Load and preprocess the Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target
    feature_names = list(data.feature_names)
    class_names   = list(data.target_names)      # ['malignant', 'benign']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, class_names, scaler


def load_diabetes_data():
    """Load and preprocess the Pima Diabetes dataset (binary: diabetic / healthy)."""
    # Using sklearn's regression dataset; binarise at median for demo
    data = load_diabetes()
    X, y = data.data.astype(np.float32), data.target
    y_bin = (y > np.median(y)).astype(int)
    feature_names = list(data.feature_names)
    class_names   = ["healthy", "diabetic"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_bin, feature_names, class_names, scaler


# ─────────────────────────────────────────────────────────────────────────────
# HEART DISEASE SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_heart_disease_data(n_samples: int = 1000, seed: int = 42):
    """Generate a synthetic heart disease dataset with clinically realistic features."""
    rng = np.random.default_rng(seed)
    feature_names = [
        "age", "sex", "chest_pain_type", "resting_bp",
        "cholesterol", "fasting_blood_sugar", "resting_ecg",
        "max_heart_rate", "exercise_angina", "st_depression",
        "st_slope", "ca_vessels", "thal"
    ]
    age         = rng.integers(30, 80, n_samples).astype(float)
    sex         = rng.integers(0, 2, n_samples).astype(float)
    chest_pain  = rng.integers(0, 4, n_samples).astype(float)
    bp          = rng.normal(130, 20, n_samples)
    cholesterol = rng.normal(240, 50, n_samples)
    fbs         = (rng.random(n_samples) > 0.85).astype(float)
    ecg         = rng.integers(0, 3, n_samples).astype(float)
    max_hr      = rng.normal(150, 25, n_samples)
    angina      = (rng.random(n_samples) > 0.7).astype(float)
    st_dep      = rng.uniform(0, 5, n_samples)
    st_slope    = rng.integers(0, 3, n_samples).astype(float)
    ca          = rng.integers(0, 4, n_samples).astype(float)
    thal        = rng.integers(0, 3, n_samples).astype(float)

    X = np.column_stack([age, sex, chest_pain, bp, cholesterol, fbs, ecg,
                         max_hr, angina, st_dep, st_slope, ca, thal])

    # Simplified risk score
    risk = (0.3*(age/80) + 0.2*(bp/200) + 0.1*angina +
            0.15*st_dep/5 + 0.15*ca/3 + 0.1*(cholesterol/400))
    y = (risk + rng.normal(0, 0.05, n_samples) > 0.35).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, ["no_disease", "heart_disease"], scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

class HealthcareXAI:
    """
    Healthcare-specific XAI pipeline.

    Workflow:
    1. Train a deep learning model on clinical data
    2. Evaluate model performance
    3. Generate SHAP + LIME explanations for individual predictions
    4. Produce compliance-ready HTML reports
    """

    def __init__(self, dataset: str = "heart_disease"):
        self.dataset = dataset
        self.model   = None
        self.engine  = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_names = self.class_names = None
        self.scaler  = None

    def load_data(self):
        logger.info(f"Loading dataset: {self.dataset}")
        if self.dataset == "breast_cancer":
            X, y, fn, cn, sc = load_breast_cancer_data()
        elif self.dataset == "diabetes":
            X, y, fn, cn, sc = load_diabetes_data()
        else:
            X, y, fn, cn, sc = generate_heart_disease_data()

        self.feature_names = fn
        self.class_names   = cn
        self.scaler        = sc
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        logger.info(f"Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        return self

    def train(self, model_type: str = "mlp"):
        logger.info(f"Training {model_type.upper()} model for healthcare...")
        self.model = get_model(model_type, epochs=60, lr=5e-4, verbose=True)
        self.model.fit(self.X_train, self.y_train)

        y_pred  = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)

        logger.info("\n" + classification_report(self.y_test, y_pred,
                                                   target_names=self.class_names))
        auc = roc_auc_score(self.y_test, y_proba[:, 1])
        logger.info(f"ROC-AUC: {auc:.4f}")
        return self

    def build_engine(self, shap_type: str = "kernel"):
        logger.info("Building XAI Engine...")
        self.engine = XAIEngine(
            model=self.model,
            X_train=self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            shap_type=shap_type,
        )
        return self

    def explain_patient(self, patient_idx: int = 0, save_html: bool = True):
        """
        Generate a clinical explanation for a single patient prediction.
        This can be shown to a clinician to support decision-making.
        """
        patient = self.X_test[patient_idx]
        true_label = self.class_names[self.y_test[patient_idx]]
        logger.info(f"Explaining patient {patient_idx} | True label: {true_label}")

        report = self.engine.explain(patient)

        # Print clinical narrative
        print("=" * 60)
        print("CLINICAL DECISION SUPPORT REPORT")
        print("=" * 60)
        print(f"Patient Index : {patient_idx}")
        print(f"True Diagnosis: {true_label}")
        print(f"AI Prediction : {report['prediction']}")
        print(f"Confidence    : {report['confidence']*100:.1f}%")
        print("-" * 60)
        print(report['narrative'])
        print("=" * 60)

        if save_html:
            path = f"outputs/healthcare_patient_{patient_idx}_report.html"
            os.makedirs("outputs", exist_ok=True)
            self.engine.generate_report(report, save_path=path)
            logger.success(f"Clinical XAI report saved: {path}")

        return report

    def global_risk_factors(self):
        """Show which clinical features are most predictive globally (SHAP)."""
        logger.info("Computing global risk factor importance...")
        global_exp = self.engine.global_explanation(self.X_test)
        print("\nTop Global Risk Factors:")
        print(global_exp["feature_importance"].head(10).to_string(index=False))
        return global_exp

    def run(self):
        """Full end-to-end healthcare XAI pipeline."""
        self.load_data().train().build_engine()
        self.explain_patient(patient_idx=0)
        self.global_risk_factors()
        logger.success("Healthcare XAI pipeline complete.")


if __name__ == "__main__":
    pipeline = HealthcareXAI(dataset="heart_disease")
    pipeline.run()
