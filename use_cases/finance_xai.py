"""
finance_xai.py - XAI for Financial Applications
=================================================
Demonstrates Explainable AI for financial decision-making.

Use Cases:
1. Credit Risk Assessment - explain why a loan was approved or denied
2. Fraud Detection        - highlight suspicious transaction patterns
3. Market Risk Prediction - explain volatility predictions

Regulatory Context:
- EU GDPR Article 22: Right to explanation for automated decisions
- ECOA / Fair Housing Act: Bias detection in credit decisions
- Basel III: Model risk management requires interpretable models

Author: Pranay M Mahendrakar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score
from loguru import logger
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_engine import XAIEngine
from deep_model import get_model


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def generate_credit_risk_data(n_samples: int = 2000, seed: int = 42):
    """Generate a realistic synthetic credit risk dataset."""
    rng = np.random.default_rng(seed)
    feature_names = [
        "credit_score", "annual_income", "debt_to_income",
        "loan_amount", "loan_tenure_months", "employment_years",
        "num_credit_lines", "num_delinquencies", "home_ownership",
        "purpose_code"
    ]
    credit_score   = rng.normal(680, 80, n_samples).clip(300, 850)
    income         = rng.lognormal(10.8, 0.6, n_samples)
    dti            = rng.beta(2, 5, n_samples) * 60
    loan_amount    = rng.lognormal(10.2, 0.7, n_samples)
    tenure         = rng.choice([12, 24, 36, 48, 60], n_samples).astype(float)
    emp_years      = rng.gamma(3, 2, n_samples).clip(0, 30)
    credit_lines   = rng.integers(1, 20, n_samples).astype(float)
    delinquencies  = rng.integers(0, 6, n_samples).astype(float)
    home_own       = rng.integers(0, 3, n_samples).astype(float)
    purpose        = rng.integers(0, 8, n_samples).astype(float)

    X = np.column_stack([credit_score, income, dti, loan_amount, tenure,
                         emp_years, credit_lines, delinquencies, home_own, purpose])

    # Default probability
    prob = (1 - credit_score/850)*0.4 + dti/60*0.3 + delinquencies/5*0.2 + 0.1*rng.random(n_samples)
    y = (prob + rng.normal(0, 0.05, n_samples) > 0.45).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, ["approved", "defaulted"], scaler


def generate_fraud_detection_data(n_samples: int = 5000, fraud_rate: float = 0.02,
                                   seed: int = 42):
    """
    Generate a synthetic credit card transaction fraud dataset.
    Highly imbalanced: ~2% fraud rate (realistic).
    """
    rng = np.random.default_rng(seed)
    feature_names = [
        "transaction_amount", "merchant_category", "hour_of_day",
        "day_of_week", "country_code", "distance_from_home",
        "velocity_1h", "velocity_24h", "card_present",
        "mcc_risk_score", "account_age_days", "avg_transaction_amount"
    ]
    n_fraud  = int(n_samples * fraud_rate)
    n_legit  = n_samples - n_fraud

    # Legitimate transactions
    legit = np.column_stack([
        rng.lognormal(4.0, 1.0, n_legit),       # amount
        rng.integers(0, 20, n_legit),             # merchant_cat
        rng.integers(8, 22, n_legit).astype(float),  # hour (business hours)
        rng.integers(0, 5, n_legit).astype(float),   # weekday
        rng.integers(0, 3, n_legit).astype(float),   # country (local)
        rng.lognormal(2, 0.5, n_legit),          # distance (short)
        rng.integers(0, 3, n_legit).astype(float),
        rng.integers(0, 10, n_legit).astype(float),
        rng.binomial(1, 0.85, n_legit).astype(float),
        rng.uniform(0, 0.3, n_legit),
        rng.integers(30, 3650, n_legit).astype(float),
        rng.lognormal(4.0, 0.5, n_legit),
    ])

    # Fraudulent transactions (different distribution)
    fraud = np.column_stack([
        rng.lognormal(5.5, 1.5, n_fraud),        # higher amounts
        rng.integers(0, 20, n_fraud),
        rng.choice([1, 2, 3, 22, 23, 0], n_fraud).astype(float),  # odd hours
        rng.integers(0, 7, n_fraud).astype(float),
        rng.integers(3, 10, n_fraud).astype(float),  # foreign countries
        rng.lognormal(5, 1.5, n_fraud),           # large distance
        rng.integers(5, 20, n_fraud).astype(float),
        rng.integers(10, 50, n_fraud).astype(float),
        rng.binomial(1, 0.2, n_fraud).astype(float),  # card not present
        rng.uniform(0.5, 1.0, n_fraud),
        rng.integers(1, 60, n_fraud).astype(float),   # new accounts
        rng.lognormal(3.5, 0.5, n_fraud),
    ])

    X = np.vstack([legit, fraud])
    y = np.hstack([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)
    shuffle = rng.permutation(n_samples)
    X, y = X[shuffle], y[shuffle]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names, ["legitimate", "fraud"], scaler


# ─────────────────────────────────────────────────────────────────────────────
# FINANCE XAI PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class FinanceXAI:
    """
    Finance-domain XAI pipeline.

    Provides transparent explanations for:
    - Credit risk decisions (with bias detection)
    - Fraud alert explanations (for analysts)
    """

    def __init__(self, use_case: str = "credit_risk"):
        self.use_case = use_case
        self.model = self.engine = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_names = self.class_names = None

    def load_data(self):
        logger.info(f"Loading {self.use_case} data...")
        if self.use_case == "credit_risk":
            X, y, fn, cn, sc = generate_credit_risk_data()
        else:
            X, y, fn, cn, sc = generate_fraud_detection_data()

        self.feature_names = fn
        self.class_names   = cn
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X, y, test_size=0.2, random_state=42)
        logger.info(f"Positive rate in test: {self.y_test.mean():.3f}")
        return self

    def train(self, model_type: str = "mlp"):
        logger.info(f"Training {model_type.upper()} for {self.use_case}...")
        self.model = get_model(model_type, epochs=80, lr=5e-4, verbose=True)
        self.model.fit(self.X_train, self.y_train)

        y_pred  = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)
        logger.info("\n" + classification_report(self.y_test, y_pred,
                                                   target_names=self.class_names))
        ap = average_precision_score(self.y_test, y_proba[:, 1])
        logger.info(f"Average Precision (PR-AUC): {ap:.4f}")
        return self

    def build_engine(self):
        self.engine = XAIEngine(
            model=self.model, X_train=self.X_train,
            feature_names=self.feature_names, class_names=self.class_names,
            shap_type="kernel")
        return self

    def explain_decision(self, applicant_idx: int = 0):
        """
        Explain a single credit/fraud decision to a compliance officer.
        This satisfies GDPR Article 22 'right to explanation'.
        """
        instance   = self.X_test[applicant_idx]
        true_label = self.class_names[self.y_test[applicant_idx]]
        report     = self.engine.explain(instance)

        print("=" * 60)
        print(f"FINANCIAL DECISION EXPLANATION  [{self.use_case.upper()}]")
        print("=" * 60)
        print(f"True outcome   : {true_label}")
        print(f"Model decision : {report['prediction']}")
        print(f"Confidence     : {report['confidence']*100:.1f}%")
        print("-" * 60)
        print(report["narrative"])

        # GDPR-style explanation summary
        top_features = report["shap_importance"].head(5)
        print("\nTop decision factors (for regulatory disclosure):")
        print(top_features.to_string(index=False))
        print("=" * 60)

        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/finance_{self.use_case}_{applicant_idx}_report.html"
        self.engine.generate_report(report, save_path=path)
        return report

    def bias_analysis(self):
        """
        Examine if any protected attribute has disproportionate SHAP contribution.
        """
        logger.info("Running bias analysis via global SHAP...")
        global_exp = self.engine.global_explanation(self.X_test[:200])
        importance = global_exp["feature_importance"]
        print("\nGlobal Feature Importance (Bias Audit):")
        print(importance.head(10).to_string(index=False))
        return importance

    def run(self):
        self.load_data().train().build_engine()
        self.explain_decision(0)
        self.bias_analysis()
        logger.success(f"Finance XAI pipeline ({self.use_case}) complete.")


if __name__ == "__main__":
    # Run credit risk demo
    FinanceXAI(use_case="credit_risk").run()
    # Run fraud detection demo
    FinanceXAI(use_case="fraud_detection").run()
