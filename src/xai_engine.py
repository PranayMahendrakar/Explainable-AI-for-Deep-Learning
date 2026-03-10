"""
xai_engine.py - Core Explainable AI Engine
==========================================
Unified XAI engine integrating SHAP and LIME for Deep Learning model interpretability.
Supports tabular, image, and text data with multiple explanation strategies.

Author: Pranay M Mahendrakar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import lime.lime_image
import lime.lime_text
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASE EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class BaseExplainer:
    """Abstract base class for all XAI explainers."""

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        logger.info(f"Initialized {self.__class__.__name__}")

    def explain(self, X: np.ndarray, **kwargs) -> Dict:
        raise NotImplementedError("Subclasses must implement explain()")

    def plot(self, explanation: Dict, **kwargs):
        raise NotImplementedError("Subclasses must implement plot()")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer(BaseExplainer):
    """
    SHAP (SHapley Additive exPlanations) wrapper.

    Supports:
    - TreeExplainer     -> gradient boosted trees
    - DeepExplainer     -> neural networks (TF/PyTorch)
    - KernelExplainer   -> model-agnostic (slower)
    - GradientExplainer -> deep learning with gradients
    """

    EXPLAINER_MAP = {
        "tree": shap.TreeExplainer,
        "deep": shap.DeepExplainer,
        "kernel": shap.KernelExplainer,
        "gradient": shap.GradientExplainer,
    }

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        explainer_type: str = "kernel",
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__(model, feature_names)
        self.background_data = background_data
        self.explainer_type = explainer_type
        self._explainer = None
        self._build_explainer()

    def _build_explainer(self):
        """Instantiate the correct SHAP explainer."""
        logger.info(f"Building SHAP {self.explainer_type} explainer...")
        cls = self.EXPLAINER_MAP.get(self.explainer_type)
        if cls is None:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")

        if self.explainer_type in ("deep", "gradient"):
            self._explainer = cls(self.model, self.background_data)
        elif self.explainer_type == "kernel":
            predict_fn = (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict
            )
            bg = (
                shap.kmeans(self.background_data, 50)
                if self.background_data is not None
                else shap.kmeans(np.zeros((1, 10)), 1)
            )
            self._explainer = cls(predict_fn, bg)
        else:
            self._explainer = cls(self.model)

    def explain(self, X: np.ndarray, **kwargs) -> Dict:
        """Compute SHAP values for input X."""
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self._explainer.shap_values(X, **kwargs)
        expected_value = self._explainer.expected_value
        return {
            "shap_values": shap_values,
            "expected_value": expected_value,
            "X": X,
            "feature_names": self.feature_names,
        }

    def plot(self, explanation: Dict, plot_type: str = "summary", **kwargs):
        """Visualise SHAP explanations. plot_type: summary|bar|waterfall|force"""
        sv   = explanation["shap_values"]
        X    = explanation["X"]
        feat = explanation.get("feature_names")
        df_X = pd.DataFrame(X, columns=feat) if feat is not None else X
        plt.figure(figsize=(12, 6))
        if plot_type == "summary":
            shap.summary_plot(sv, df_X, show=False)
        elif plot_type == "bar":
            shap.summary_plot(sv, df_X, plot_type="bar", show=False)
        elif plot_type == "waterfall":
            ev      = explanation["expected_value"]
            ev_val  = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
            sv_row  = sv[0] if isinstance(sv, list) else sv[0]
            shap.waterfall_plot(
                shap.Explanation(values=sv_row, base_values=ev_val,
                                 data=X[0], feature_names=feat), show=False)
        elif plot_type == "force":
            ev     = explanation["expected_value"]
            ev_val = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
            sv_row = sv[0] if isinstance(sv, list) else sv[0]
            shap.force_plot(ev_val, sv_row, X[0], feature_names=feat,
                            matplotlib=True, show=False)
        plt.tight_layout()
        return plt.gcf()

    def get_feature_importance(self, explanation: Dict) -> pd.DataFrame:
        """Return mean absolute SHAP values as a feature importance table."""
        sv = explanation["shap_values"]
        if isinstance(sv, list):
            sv = np.abs(np.array(sv)).mean(axis=0)
        mean_abs = np.abs(sv).mean(axis=0)
        names = explanation.get("feature_names") or [
            f"feature_{i}" for i in range(len(mean_abs))
        ]
        df = pd.DataFrame({"feature": names, "importance": mean_abs})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIME EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

class LIMEExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) wrapper.
    Supports tabular, image, and text explanations.
    """

    def __init__(
        self,
        model: Any,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "tabular",
        categorical_features: Optional[List[int]] = None,
    ):
        super().__init__(model, feature_names)
        self.training_data    = training_data
        self.class_names      = class_names
        self.mode             = mode
        self.categorical_features = categorical_features or []
        self._explainer       = None
        self._build_explainer()

    def _build_explainer(self):
        logger.info(f"Building LIME {self.mode} explainer...")
        if self.mode == "tabular":
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.training_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                categorical_features=self.categorical_features,
                mode="classification",
                random_state=42,
            )
        elif self.mode == "image":
            self._explainer = lime.lime_image.LimeImageExplainer(random_state=42)
        elif self.mode == "text":
            self._explainer = lime.lime_text.LimeTextExplainer(
                class_names=self.class_names, random_state=42)
        else:
            raise ValueError(f"Unsupported LIME mode: {self.mode}")

    def explain(self, X, num_features: int = 10, num_samples: int = 1000,
                **kwargs) -> Dict:
        """Generate a LIME explanation for a single instance."""
        logger.info(f"Generating LIME {self.mode} explanation...")
        predict_fn = (self.model.predict_proba
                      if hasattr(self.model, "predict_proba")
                      else self.model.predict)
        if self.mode == "tabular":
            exp = self._explainer.explain_instance(
                X, predict_fn, num_features=num_features,
                num_samples=num_samples, **kwargs)
        elif self.mode == "image":
            exp = self._explainer.explain_instance(
                X, predict_fn, top_labels=1, num_samples=num_samples, **kwargs)
        elif self.mode == "text":
            exp = self._explainer.explain_instance(
                X, predict_fn, num_features=num_features, **kwargs)
        return {"lime_exp": exp, "mode": self.mode}

    def plot(self, explanation: Dict, **kwargs):
        exp = explanation["lime_exp"]
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        return fig

    def get_feature_importance(self, explanation: Dict) -> pd.DataFrame:
        items = explanation["lime_exp"].as_list()
        df = pd.DataFrame(items, columns=["feature", "importance"])
        df["abs_importance"] = df["importance"].abs()
        return df.sort_values("abs_importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNIFIED XAI ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class XAIEngine:
    """
    Unified Explainable AI Engine.
    Combines SHAP + LIME for comprehensive deep-learning interpretability.

    Usage:
        engine = XAIEngine(model, X_train, feature_names, class_names)
        report = engine.explain(X_test[0])
        engine.generate_report(report, "report.html")
    """

    def __init__(self, model, X_train, feature_names=None,
                 class_names=None, shap_type="kernel", task="classification"):
        self.model         = model
        self.X_train       = X_train
        self.feature_names = feature_names
        self.class_names   = class_names
        self.task          = task
        logger.info("Initialising XAI Engine with SHAP + LIME...")
        self.shap_explainer = SHAPExplainer(
            model=model, background_data=X_train,
            explainer_type=shap_type, feature_names=feature_names)
        self.lime_explainer = LIMEExplainer(
            model=model, training_data=X_train,
            feature_names=feature_names, class_names=class_names, mode="tabular")

    def explain(self, X_instance, shap_plot_type="waterfall",
                num_lime_features=10) -> Dict:
        """Full XAI report for a single prediction instance."""
        X_2d = X_instance.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba      = self.model.predict_proba(X_2d)[0]
            pred_class = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            pred_val   = self.model.predict(X_2d)[0]
            pred_class = int(pred_val > 0.5)
            confidence = float(pred_val)

        class_label = (self.class_names[pred_class]
                       if self.class_names and pred_class < len(self.class_names)
                       else str(pred_class))
        logger.info(f"Prediction: {class_label} (confidence={confidence:.3f})")

        shap_exp        = self.shap_explainer.explain(X_2d)
        shap_fig        = self.shap_explainer.plot(shap_exp, plot_type=shap_plot_type)
        shap_importance = self.shap_explainer.get_feature_importance(shap_exp)

        lime_exp        = self.lime_explainer.explain(X_instance, num_features=num_lime_features)
        lime_fig        = self.lime_explainer.plot(lime_exp)
        lime_importance = self.lime_explainer.get_feature_importance(lime_exp)

        narrative = self._generate_narrative(class_label, confidence,
                                             shap_importance, lime_importance)
        return {
            "prediction": class_label, "predicted_class_index": pred_class,
            "confidence": confidence,
            "shap_explanation": shap_exp, "shap_figure": shap_fig,
            "shap_importance": shap_importance,
            "lime_explanation": lime_exp, "lime_figure": lime_fig,
            "lime_importance": lime_importance,
            "narrative": narrative,
        }

    def _generate_narrative(self, prediction, confidence, shap_df, lime_df) -> str:
        top_shap = shap_df.head(3)["feature"].tolist()
        top_lime = lime_df.head(3)["feature"].tolist()
        overlap  = set(top_shap) & set(top_lime)
        return (
            f"The model predicted '{prediction}' with {confidence*100:.1f}% confidence.\n\n"
            f"[SHAP] Most influential features: {', '.join(top_shap)}.\n"
            f"SHAP values represent each feature's marginal contribution to the prediction.\n\n"
            f"[LIME] Locally important factors: {', '.join(top_lime)}.\n"
            f"LIME fits a local linear model to approximate the decision boundary.\n\n"
            f"Both SHAP and LIME agree on: {overlap if overlap else 'partially overlapping features'}."
        )

    def batch_explain(self, X, n_samples=5):
        return [self.explain(X[i]) for i in range(min(n_samples, len(X)))]

    def global_explanation(self, X) -> Dict:
        """Compute global feature importance via SHAP across the dataset."""
        logger.info("Computing global SHAP explanation...")
        shap_exp   = self.shap_explainer.explain(X)
        global_fig = self.shap_explainer.plot(shap_exp, plot_type="summary")
        bar_fig    = self.shap_explainer.plot(shap_exp, plot_type="bar")
        importance = self.shap_explainer.get_feature_importance(shap_exp)
        return {"shap_values": shap_exp["shap_values"],
                "global_summary_figure": global_fig,
                "global_bar_figure": bar_fig,
                "feature_importance": importance}

    def generate_report(self, explanation: Dict, save_path: str = "xai_report.html"):
        """Save a self-contained HTML XAI report."""
        import base64, io

        def fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        shap_b64   = fig_to_b64(explanation["shap_figure"])
        lime_b64   = fig_to_b64(explanation["lime_figure"])
        shap_table = explanation["shap_importance"].head(10).to_html(index=False)
        lime_table = explanation["lime_importance"].head(10).to_html(index=False)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>XAI Report</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;max-width:1100px;margin:auto;padding:2rem}}
h1{{color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:.5rem}}
h2{{color:#79c0ff;margin-top:2rem}}
.badge{{display:inline-block;background:#388bfd22;color:#79c0ff;border:1px solid #388bfd;
        border-radius:20px;padding:.3rem 1rem;margin:.2rem;font-size:.9rem}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1.5rem;margin:1rem 0}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:#21262d;padding:.5rem;text-align:left}}
td{{border-bottom:1px solid #30363d;padding:.4rem}}
img{{max-width:100%;border-radius:6px;margin:1rem 0}}
pre{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:1rem;white-space:pre-wrap}}
</style></head>
<body>
<h1>Explainable AI Report</h1>
<div class="card">
  <h2>Prediction Summary</h2>
  <span class="badge">Prediction: {explanation['prediction']}</span>
  <span class="badge">Confidence: {explanation['confidence']*100:.1f}%</span>
</div>
<div class="card">
  <h2>Plain-Language Explanation</h2>
  <pre>{explanation['narrative']}</pre>
</div>
<div class="card">
  <h2>SHAP Explanation</h2>
  <img src="data:image/png;base64,{shap_b64}" alt="SHAP Plot">
  <h3>Top SHAP Features</h3>{shap_table}
</div>
<div class="card">
  <h2>LIME Explanation</h2>
  <img src="data:image/png;base64,{lime_b64}" alt="LIME Plot">
  <h3>Top LIME Features</h3>{lime_table}
</div>
</body></html>"""

        with open(save_path, "w") as f:
            f.write(html)
        logger.success(f"XAI report saved to {save_path}")
        return save_path
