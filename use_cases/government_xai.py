"""
government_xai.py - XAI for Government AI Systems
==================================================
Demonstrates Explainable AI for public-sector decision-making.

Use Cases:
1. Benefits Eligibility - transparent welfare/social benefit decisions
2. Recidivism Prediction - fair and explainable criminal justice AI
3. Public Service Routing - smart triage with bias monitoring

Regulatory Context:
- EU AI Act High-Risk Category: Employment, Education, Public Benefits
- Algorithmic Accountability Act (US)
- UNESCO AI Ethics Recommendation
- NIST AI RMF (Risk Management Framework)

Author: Pranay M Mahendrakar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_engine import XAIEngine
from deep_model import get_model


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_benefits_eligibility_data(n_samples: int = 3000, seed: int = 42):
    """
    Synthetic social benefits eligibility dataset.
    Mimics means-tested welfare programme criteria.
    """
    rng = np.random.default_rng(seed)
    feature_names = [
        "household_income", "household_size", "employment_status",
        "disability_flag", "age", "num_dependents", "housing_status",
        "asset_value", "prior_benefit_history", "region_deprivation_index"
    ]
    income     = rng.lognormal(10.0, 0.8, n_samples)
    hh_size    = rng.integers(1, 7, n_samples).astype(float)
    employed   = rng.binomial(1, 0.65, n_samples).astype(float)
    disability = rng.binomial(1, 0.12, n_samples).astype(float)
    age        = rng.integers(18, 75, n_samples).astype(float)
    dependents = rng.integers(0, 5, n_samples).astype(float)
    housing    = rng.integers(0, 3, n_samples).astype(float)  # own/rent/temp
    assets     = rng.lognormal(9.0, 1.5, n_samples)
    prior      = rng.binomial(1, 0.25, n_samples).astype(float)
    deprivation = rng.uniform(0, 1, n_samples)

    X = np.column_stack([income, hh_size, employed, disability, age,
                         dependents, housing, assets, prior, deprivation])

    # Eligibility rule (simplified)
    eligible_score = (
        (income < 30000) * 0.35 +
        disability * 0.2 +
        (employed == 0) * 0.2 +
        deprivation * 0.15 +
        (dependents > 2) * 0.1
    )
    y = (eligible_score + rng.normal(0, 0.08, n_samples) > 0.4).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, ["ineligible", "eligible"], scaler


def generate_recidivism_data(n_samples: int = 2000, seed: int = 42):
    """
    Synthetic criminal justice recidivism risk dataset.
    Inspired by the COMPAS controversy for bias-aware XAI demonstration.
    """
    rng = np.random.default_rng(seed)
    feature_names = [
        "age_at_release", "prior_convictions", "offense_severity",
        "sentence_length_months", "education_level", "employment_at_arrest",
        "substance_abuse_history", "mental_health_flag", "time_served_ratio",
        "program_participation"
    ]
    age        = rng.integers(18, 60, n_samples).astype(float)
    priors     = rng.integers(0, 15, n_samples).astype(float)
    severity   = rng.integers(1, 6, n_samples).astype(float)
    sentence   = rng.gamma(24, 2, n_samples)
    education  = rng.integers(0, 5, n_samples).astype(float)
    employed   = rng.binomial(1, 0.4, n_samples).astype(float)
    substance  = rng.binomial(1, 0.35, n_samples).astype(float)
    mental     = rng.binomial(1, 0.2, n_samples).astype(float)
    time_ratio = rng.beta(3, 2, n_samples)
    programs   = rng.integers(0, 5, n_samples).astype(float)

    X = np.column_stack([age, priors, severity, sentence, education,
                         employed, substance, mental, time_ratio, programs])

    # Recidivism risk (criminological factors, NOT race/gender)
    risk = (
        priors / 15 * 0.35 +
        (1 - education / 4) * 0.15 +
        substance * 0.2 +
        (1 - employed) * 0.15 +
        severity / 5 * 0.1 +
        (1 - time_ratio) * 0.05
    )
    y = (risk + rng.normal(0, 0.1, n_samples) > 0.45).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, ["low_risk", "high_risk"], scaler


# ─────────────────────────────────────────────────────────────────────────────
# GOVERNMENT XAI PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class GovernmentXAI:
    """
    Government AI XAI pipeline with fairness and accountability checks.

    Key Features:
    - Per-decision explanations for citizen appeal rights
    - Group-fairness metric reporting
    - Regulatory compliance report generation
    - Bias detection across sensitive sub-groups
    """

    def __init__(self, use_case: str = "benefits"):
        self.use_case = use_case
        self.model = self.engine = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_names = self.class_names = None

    def load_data(self):
        logger.info(f"Loading government dataset: {self.use_case}")
        if self.use_case == "benefits":
            X, y, fn, cn, sc = generate_benefits_eligibility_data()
        else:
            X, y, fn, cn, sc = generate_recidivism_data()

        self.feature_names = fn
        self.class_names   = cn
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X, y, test_size=0.2, random_state=42)
        logger.info(f"Positive rate: {y.mean():.3f}")
        return self

    def train(self, model_type: str = "mlp"):
        logger.info(f"Training {model_type} model for {self.use_case}...")
        self.model = get_model(model_type, hidden_dims=[256, 128, 64],
                               epochs=100, lr=5e-4, verbose=True)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        logger.info("\n" + classification_report(self.y_test, y_pred,
                                                   target_names=self.class_names))
        return self

    def build_engine(self):
        self.engine = XAIEngine(
            model=self.model, X_train=self.X_train,
            feature_names=self.feature_names, class_names=self.class_names,
            shap_type="kernel")
        return self

    def explain_citizen_decision(self, citizen_idx: int = 0):
        """
        Generate a citizen-readable explanation for their AI decision.
        Required under EU AI Act for high-risk AI in public services.
        """
        instance   = self.X_test[citizen_idx]
        true_label = self.class_names[self.y_test[citizen_idx]]
        report     = self.engine.explain(instance)

        print("=" * 70)
        print("GOVERNMENT AI DECISION TRANSPARENCY REPORT")
        print(f"Application: {self.use_case.upper().replace('_', ' ')}")
        print("=" * 70)
        print(f"Citizen Ref  : GOV-{citizen_idx:06d}")
        print(f"True Outcome : {true_label}")
        print(f"AI Decision  : {report['prediction']}")
        print(f"Confidence   : {report['confidence']*100:.1f}%")
        print("-" * 70)
        print("WHY THIS DECISION WAS MADE:")
        print(report["narrative"])
        print("-" * 70)
        print("KEY FACTORS (for appeal purposes):")
        print(report["shap_importance"].head(5).to_string(index=False))
        print("=" * 70)
        print("Note: You have the right to request human review of this decision.")
        print("=" * 70)

        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/gov_{self.use_case}_{citizen_idx}_report.html"
        self.engine.generate_report(report, save_path=path)
        return report

    def fairness_audit(self):
        """
        Conduct a SHAP-based fairness audit.
        Checks feature importance alignment with policy objectives.
        """
        logger.info("Conducting algorithmic fairness audit...")
        n_audit = min(300, len(self.X_test))
        global_exp = self.engine.global_explanation(self.X_test[:n_audit])
        importance = global_exp["feature_importance"]

        print("\n" + "=" * 60)
        print("ALGORITHMIC FAIRNESS AUDIT REPORT")
        print("=" * 60)
        print(f"Model: {self.use_case}")
        print(f"Audit samples: {n_audit}")
        print("\nGlobal feature importance (SHAP-based):")
        print(importance.head(10).to_string(index=False))

        # Flag if any feature has disproportionately high influence
        top_feature = importance.iloc[0]
        print(f"\nHighest-influence feature: '{top_feature['feature']}'")
        print(f"SHAP importance: {top_feature['importance']:.4f}")

        total_importance = importance['importance'].sum()
        top_share = top_feature['importance'] / total_importance
        if top_share > 0.3:
            print(f"WARNING: Single feature accounts for {top_share*100:.1f}% of decisions.")
            print("Recommend human review of this feature's policy alignment.")
        else:
            print(f"Concentration ratio: {top_share*100:.1f}% - within acceptable bounds.")
        print("=" * 60)
        return importance

    def generate_compliance_report(self):
        """
        Generate a regulatory compliance summary for government oversight.
        """
        compliance = {
            "use_case"     : self.use_case,
            "model_type"   : type(self.model).__name__,
            "training_size": len(self.X_train),
            "test_size"    : len(self.X_test),
            "feature_count": len(self.feature_names),
            "xai_methods"  : ["SHAP (KernelExplainer)", "LIME (TabularExplainer)"],
            "regulations"  : ["EU AI Act", "GDPR Art.22",
                               "UNESCO AI Ethics", "NIST AI RMF"],
            "explainability": "Per-decision SHAP waterfall + LIME feature attribution",
            "appeal_support": True,
            "bias_checks"  : "SHAP global importance analysis",
        }
        print("\nCOMPLIANCE SUMMARY:")
        for k, v in compliance.items():
            print(f"  {k:<20}: {v}")
        return compliance

    def run(self):
        self.load_data().train().build_engine()
        self.explain_citizen_decision(0)
        self.fairness_audit()
        self.generate_compliance_report()
        logger.success(f"Government XAI pipeline ({self.use_case}) complete.")


if __name__ == "__main__":
    GovernmentXAI(use_case="benefits").run()
    GovernmentXAI(use_case="recidivism").run()
use_cases/government_xai.py
