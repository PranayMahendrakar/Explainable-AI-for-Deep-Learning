"""dashboard.py - Streamlit XAI Interactive Dashboard
Run: streamlit run app/dashboard.py
Author: Pranay M Mahendrakar
"""
import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "use_cases"))

st.set_page_config(page_title="Explainable AI Dashboard", page_icon="🧠", layout="wide")

@st.cache_resource
def load_modules():
    from xai_engine import XAIEngine
    from deep_model import get_model
    from healthcare_xai import generate_heart_disease_data
    from finance_xai import generate_credit_risk_data, generate_fraud_detection_data
    from government_xai import generate_benefits_eligibility_data, generate_recidivism_data
    return (XAIEngine, get_model, generate_heart_disease_data,
            generate_credit_risk_data, generate_fraud_detection_data,
            generate_benefits_eligibility_data, generate_recidivism_data)

with st.sidebar:
    st.title("🧠 XAI Control Panel")
    domain     = st.selectbox("Domain", ["Healthcare - Heart Disease",
                                          "Finance - Credit Risk",
                                          "Finance - Fraud Detection",
                                          "Government - Benefits",
                                          "Government - Recidivism"])
    model_type = st.selectbox("Model", ["mlp", "cnn", "lstm", "transformer"])
    epochs     = st.slider("Epochs", 10, 200, 60, 10)
    shap_type  = st.selectbox("SHAP Type", ["kernel", "tree", "gradient"])
    n_explain  = st.slider("Instances to explain", 1, 10, 3)
    shap_plot  = st.selectbox("SHAP Plot", ["waterfall", "force", "summary", "bar"])
    run_btn    = st.button("Train & Explain", type="primary", use_container_width=True)

st.title("🧠 Explainable AI for Deep Learning")
st.caption("Transparency | Trust | Accountability - powered by SHAP & LIME")
tabs = st.tabs(["Overview", "Train & Explain", "Global Insights", "Batch Analysis"])

with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.metric("XAI Methods", "SHAP + LIME")
    col2.metric("Model Zoo", "MLP | CNN | LSTM | Transformer")
    col3.metric("Domains", "Healthcare | Finance | Government")
    st.markdown("### Regulatory Alignment")
    st.dataframe(pd.DataFrame({
        "Domain":     ["Healthcare", "Finance", "Government"],
        "Regulation": ["FDA SaMD, EU AI Act", "GDPR Art.22, ECOA", "EU AI Act, NIST RMF"],
        "XAI Need":   ["Clinical transparency", "Right to explanation", "Human oversight"],
    }), use_container_width=True)

with tabs[1]:
    if run_btn:
        (XAIEngine, get_model, gen_heart, gen_credit,
         gen_fraud, gen_benefits, gen_recid) = load_modules()
        domain_map = {
            "Healthcare - Heart Disease": gen_heart,
            "Finance - Credit Risk":      gen_credit,
            "Finance - Fraud Detection":  gen_fraud,
            "Government - Benefits":      gen_benefits,
            "Government - Recidivism":    gen_recid,
        }
        with st.spinner("Generating dataset..."):
            X, y, feature_names, class_names, scaler = domain_map[domain]()
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success(f"Loaded {len(X_train)} train / {len(X_test)} test samples")
        with st.spinner(f"Training {model_type.upper()}..."):
            model = get_model(model_type, epochs=epochs, lr=5e-4, verbose=False)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
        st.success(f"Accuracy: {acc*100:.1f}%")
        with st.spinner("Building XAI Engine..."):
            engine = XAIEngine(model=model, X_train=X_train,
                               feature_names=feature_names, class_names=class_names,
                               shap_type=shap_type)
        for idx in range(n_explain):
            with st.spinner(f"Explaining instance {idx+1}..."):
                report = engine.explain(X_test[idx], shap_plot_type=shap_plot)
            with st.expander(f"Instance {idx+1} | {report["prediction"]} ({report["confidence"]*100:.1f}%)", expanded=(idx==0)):
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("SHAP")
                    st.pyplot(report["shap_figure"])
                    st.dataframe(report["shap_importance"].head(8))
                with c2:
                    st.subheader("LIME")
                    st.pyplot(report["lime_figure"])
                    st.dataframe(report["lime_importance"].head(8))
                st.info(report["narrative"])
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                engine.generate_report(report, save_path=tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(f"Download Report #{idx+1}", f.read(), f"xai_{idx}.html", "text/html")
        st.session_state.update({"engine": engine, "X_test": X_test})
    else:
        st.info("Configure settings in the sidebar and click Train & Explain.")

with tabs[2]:
    if "engine" in st.session_state:
        n_g = st.slider("Samples", 50, 500, 200)
        if st.button("Compute Global SHAP"):
            with st.spinner("Computing..."):
                g = st.session_state["engine"].global_explanation(st.session_state["X_test"][:n_g])
            c1, c2 = st.columns(2)
            with c1: st.pyplot(g["global_summary_figure"])
            with c2: st.pyplot(g["global_bar_figure"])
            st.dataframe(g["feature_importance"])
    else:
        st.info("Run training first.")

with tabs[3]:
    uploaded = st.file_uploader("Upload CSV for batch explanation", type="csv")
    if uploaded and "engine" in st.session_state:
        df = pd.read_csv(uploaded)
        n_b = st.slider("Instances", 1, min(20, len(df)), 5)
        if st.button("Run Batch XAI"):
            results = st.session_state["engine"].batch_explain(df.values.astype("float32"), n_b)
            st.dataframe(pd.DataFrame([{
                "Instance": i, "Prediction": r["prediction"],
                "Confidence": f"{r["confidence"]*100:.1f}%"
            } for i, r in enumerate(results)]))
