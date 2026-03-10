# 🧠 Explainable AI for Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-orange)](https://shap.readthedocs.io)
[![LIME](https://img.shields.io/badge/LIME-0.2%2B-green)](https://github.com/marcotcr/lime)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.23%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **An ML system that explains *why* a prediction was made.**
> Combines SHAP + LIME with a Deep Learning model zoo for transparent, trustworthy AI across Healthcare, Finance, and Government domains.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [XAI Methods](#xai-methods)
- [Deep Learning Models](#deep-learning-models)
- [Dashboard](#dashboard)
- [Regulatory Compliance](#regulatory-compliance)
- [Research Focus](#research-focus)
- [Contributing](#contributing)

---

## 🔍 Overview

Explainable AI (XAI) bridges the gap between powerful black-box deep learning models and the **human need to understand, trust, and audit** AI decisions.

This project delivers a production-ready XAI framework that:

- Wraps any deep learning model with **SHAP** and **LIME** explanations
- Provides **per-prediction narratives** in plain English
- Generates **HTML audit reports** suitable for compliance officers
- Supports **Healthcare, Finance, and Government** AI use cases
- Implements a **fairness audit** layer for bias detection

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    XAI ENGINE (xai_engine.py)                   │
│                                                                  │
│  Input ──► Deep Learning Model ──► Prediction + Confidence      │
│                    │                                            │
│         ┌──────────┴──────────┐                                 │
│         ▼                     ▼                                 │
│   SHAP Explainer         LIME Explainer                         │
│  (Global + Local)       (Local surrogate)                       │
│         │                     │                                 │
│         └──────────┬──────────┘                                 │
│                    ▼                                            │
│           Plain-English Narrative                               │
│           HTML Report Generator                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 SHAP Explanations | Waterfall, force, summary, bar plots |
| 🟢 LIME Explanations | Local linear surrogate per prediction |
| 🌐 Global XAI | Dataset-level feature importance |
| 📄 HTML Reports | Self-contained audit reports |
| 🏥 Healthcare | Heart disease, breast cancer, diabetes |
| 💰 Finance | Credit risk, fraud detection |
| 🏛️ Government | Benefits eligibility, recidivism |
| 📊 Dashboard | Streamlit interactive web app |
| ⚖️ Fairness Audit | SHAP-based bias detection |
| 🤖 Model Zoo | MLP, CNN, LSTM, Transformer |

---

## 📁 Project Structure

```
Explainable-AI-for-Deep-Learning/
│
├── src/
│   ├── xai_engine.py          # Core XAI Engine (SHAP + LIME)
│   └── deep_model.py          # Deep Learning Model Zoo (PyTorch)
│
├── use_cases/
│   ├── healthcare_xai.py      # Healthcare: heart disease, cancer, diabetes
│   ├── finance_xai.py         # Finance: credit risk, fraud detection
│   └── government_xai.py      # Government: benefits, recidivism
│
├── app/
│   └── dashboard.py           # Streamlit interactive dashboard
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/PranayMahendrakar/Explainable-AI-for-Deep-Learning.git
cd Explainable-AI-for-Deep-Learning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a Use Case

```bash
# Healthcare - Heart Disease
python use_cases/healthcare_xai.py

# Finance - Credit Risk
python use_cases/finance_xai.py

# Government - Benefits Eligibility
python use_cases/government_xai.py
```

### 4. Launch the Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

### 5. Quick API Usage

```python
from src.xai_engine import XAIEngine
from src.deep_model  import MLPClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MLPClassifier(epochs=100, hidden_dims=[128, 64, 32])
model.fit(X_train, y_train)

# Build XAI Engine
engine = XAIEngine(
    model=model,
    X_train=X_train,
    feature_names=list(data.feature_names),
    class_names=list(data.target_names),
    shap_type="kernel"
)

# Explain a prediction
report = engine.explain(X_test[0])
print(report["narrative"])

# Save HTML report
engine.generate_report(report, "my_report.html")
```

---

## 🏥 Use Cases

### Healthcare

```python
from use_cases.healthcare_xai import HealthcareXAI
HealthcareXAI(dataset="heart_disease").run()
```

Clinical decision support with per-patient SHAP waterfall charts showing which biomarkers drove the diagnosis.

**Regulatory**: FDA AI/ML SaMD guidance | EU AI Act (High-Risk AI) | HIPAA audit trails

---

### 💰 Finance

```python
from use_cases.finance_xai import FinanceXAI
FinanceXAI(use_case="credit_risk").run()
```

Transparent credit decisions with LIME local explanations for GDPR Article 22 compliance.

**Regulatory**: GDPR Article 22 | ECOA / Fair Housing Act | Basel III Model Risk

---

### 🏛️ Government

```python
from use_cases.government_xai import GovernmentXAI
GovernmentXAI(use_case="benefits").run()
```

Citizen-readable explanations with fairness audits for public-sector AI accountability.

**Regulatory**: EU AI Act | Algorithmic Accountability Act | NIST AI RMF | UNESCO AI Ethics

---

## 🔬 XAI Methods

### SHAP (SHapley Additive exPlanations)

SHAP values fairly distribute a prediction's "credit" among input features using coalitional game theory.

| Explainer | Best For |
|-----------|----------|
| `KernelExplainer` | Any model (model-agnostic) |
| `TreeExplainer` | Gradient boosted trees |
| `DeepExplainer` | Neural networks (TF/PyTorch) |
| `GradientExplainer` | Deep learning with gradients |

**Plots**: Summary | Bar | Waterfall | Force

### LIME (Local Interpretable Model-agnostic Explanations)

LIME perturbs the input neighbourhood and fits a local linear model to approximate the black-box decision boundary.

| Mode | Input Type |
|------|-----------|
| `tabular` | Structured features |
| `image` | Image pixels |
| `text` | NLP / text data |

---

## 🤖 Deep Learning Models

| Model | Class | Use Case |
|-------|-------|----------|
| MLP | `MLPClassifier` | Tabular classification |
| CNN (1D) | `CNNClassifier` | Feature sequences |
| LSTM | `LSTMClassifier` | Time-series |
| Transformer | `TransformerClassifier` | Complex tabular |
| AutoEncoder | `AutoEncoder` | Anomaly detection |

All classifiers are **sklearn-compatible** (fit / predict / predict_proba) and **SHAP/LIME-ready**.

---

## 📊 Dashboard

The Streamlit dashboard provides a **no-code interface** for:

- Selecting domain and dataset
- Choosing model architecture and hyperparameters
- Interactive SHAP + LIME visualisation
- Downloading self-contained HTML reports
- Batch explanation with CSV upload

```bash
streamlit run app/dashboard.py
```

---

## ⚖️ Regulatory Compliance

| Regulation | Domain | Requirement Met |
|-----------|--------|----------------|
| EU AI Act (Art. 13) | All | Transparency & explainability |
| GDPR Article 22 | Finance | Right to explanation |
| FDA SaMD Guidance | Healthcare | Clinical decision transparency |
| ECOA / FHA | Finance | Bias detection + fairness audit |
| NIST AI RMF | Government | Risk management documentation |
| UNESCO AI Ethics | Government | Human oversight + appeal rights |

---

## 🔭 Research Focus

- **Model Interpretability**: Global and local feature attribution
- **Explainable Predictions**: Per-instance narratives in plain English
- **Fairness-Aware XAI**: SHAP concentration analysis for bias detection
- **Multi-Method Consensus**: SHAP + LIME agreement metrics
- **Production Deployment**: FastAPI + Streamlit ready

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-xai`)
3. Commit your changes (`git commit -m "Add amazing XAI feature"`)
4. Push to the branch (`git push origin feature/amazing-xai`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file.

---

## 👤 Author

**Pranay M Mahendrakar**

- AI Specialist | Author | Patent Holder | Open-Source Contributor
- Nodal Coordinator at IIRS-ISRO | Instructor at Tutorials Point
- [GitHub](https://github.com/PranayMahendrakar) | [Website](https://sonytech.in/pranay/)

---

*"The goal of AI is not to replace human judgment, but to augment it with transparent, accountable, and explainable intelligence."*
