import streamlit as st
import pandas as pd
import joblib
import os

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Risk Engine",
    layout="wide"
)

# ── Custom Styling ─────────────────────────────────────────
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

h1 {
    font-size: 2.2rem;
}

h2 {
    font-size: 1.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgb_classifier.pkl")
    rf_model = joblib.load("models/rf_regressor.pkl")
    scaler = joblib.load("models/scaler.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return xgb_model, rf_model, scaler, explainer

xgb_model, rf_model, scaler, explainer = load_models()

# ── Sidebar Navigation ─────────────────────────────────────
st.sidebar.title("Supply Chain Risk Engine")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Predict Shipment Risk",
        "SHAP Explainability",
        "Customer Segments"
    ]
)

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":

    st.title("Supply Chain Risk Engine")

    st.markdown("""
    An end-to-end machine learning system that predicts shipment delays,
    estimates delay severity, and segments customers by risk using
    more than 180,000 real logistics records.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Classification Model", "XGBoost")
        st.metric("AUC-ROC", "0.93")

    with col2:
        st.metric("Regression Model", "Random Forest")
        st.metric("MAE", "0.47 Days")

    with col3:
        st.metric("Customer Segmentation", "K-Means")
        st.metric("Risk Tiers", "4")

    st.markdown("---")

    st.subheader("Prediction Pipeline")

    st.markdown("""
    **New Shipment Order**

    ↓

    **Stage 1: Classification**  
    Will the shipment be delayed? (XGBoost)

    ↓

    **Stage 2: Regression**  
    Estimate delay duration in days (Random Forest)

    ↓

    **Stage 3: Customer Profiling**  
    Assign customer risk tier (K-Means)

    ↓

    **Model Explanation**  
    Explain prediction drivers using SHAP
    """)

    st.markdown("---")

    st.subheader("Business Insights")

    st.markdown("""
    - First Class shipping shows the highest late-delivery rate.
    - Customer history is the strongest predictor of future delays.
    - The regression model explains a meaningful share of delivery variance.
    - A small group of customers contributes disproportionately to delayed shipments.
    """)

# ══════════════════════════════════════════════════════════
# PAGE 2 — PREDICT SHIPMENT RISK
# ══════════════════════════════════════════════════════════
elif page == "Predict Shipment Risk":

    st.title("Shipment Delay Risk Prediction")

    st.markdown(
        "Upload a CSV file containing shipment records to generate risk predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload Shipment CSV",
        type=["csv"]
    )

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        try:
            model_features = xgb_model.get_booster().feature_names

            missing = [
                feature
                for feature in model_features
                if feature not in data.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")

            else:
                X = data[model_features]

                predictions = xgb_model.predict(X)
                probabilities = xgb_model.predict_proba(X)[:, 1]

                data["Delay Risk"] = predictions
                data["Risk Probability"] = (
                    probabilities * 100
                ).round(1)

                data["Risk Label"] = data["Risk Probability"].apply(
                    lambda x:
                    "High Risk"
                    if x > 70
                    else (
                        "Medium Risk"
                        if x > 40
                        else "Low Risk"
                    )
                )

                st.markdown("---")

                st.subheader("Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Shipments",
                        len(data)
                    )

                with col2:
                    st.metric(
                        "High Risk Shipments",
                        int((data["Delay Risk"] == 1).sum())
                    )

                with col3:
                    st.metric(
                        "Average Risk",
                        f"{data['Risk Probability'].mean():.1f}%"
                    )

                results = data[
                    ["Risk Label", "Risk Probability"]
                ].join(
                    data.drop(
                        columns=[
                            "Delay Risk",
                            "Risk Probability",
                            "Risk Label"
                        ]
                    )
                )

                st.dataframe(results)

                csv = results.to_csv(index=False)

                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="shipment_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info(
                "Ensure the uploaded CSV contains the same features used during model training."
            )

    else:
        st.info(
            "Upload a CSV file to get started. The file should contain the same features used during training."
        )

# ══════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════
elif page == "SHAP Explainability":

    st.title("SHAP Feature Explainability")

    st.markdown("""
    SHAP values explain why the model makes a prediction,
    not only which features are important but also how
    individual feature values influence the outcome.
    """)

    tab1, tab2 = st.tabs(
        ["Summary Plot", "Waterfall Plot"]
    )

    with tab1:

        st.subheader("Global Feature Importance")

        st.markdown(
            "Identify which features influence delay predictions across all shipments."
        )

        if os.path.exists(
            "reports/figures/shap_summary.png"
        ):
            st.image(
                "reports/figures/shap_summary.png",
                use_container_width=True
            )
        else:
            st.warning(
                "Run the classification notebook to generate SHAP plots."
            )

        st.markdown("""
        **How to interpret the plot**

        - Features are ranked by overall importance.
        - Each point represents a shipment.
        - Higher feature values are shown in red.
        - Lower feature values are shown in blue.
        - Points on the right increase delay risk.
        - Points on the left decrease delay risk.
        """)

    with tab2:

        st.subheader("Individual Prediction Explanation")

        st.markdown(
            "Understand why a specific shipment was classified as high risk."
        )

        if os.path.exists(
            "reports/figures/shap_waterfall.png"
        ):
            st.image(
                "reports/figures/shap_waterfall.png",
                use_container_width=True
            )
        else:
            st.warning(
                "Run the classification notebook to generate SHAP plots."
            )

        st.markdown("""
        **How to interpret the plot**

        - The base value is the average model prediction.
        - Each bar represents a feature contribution.
        - Positive contributions increase delay risk.
        - Negative contributions decrease delay risk.
        - The final value is the model's prediction.
        """)

# ══════════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════
elif page == "Customer Segments":

    st.title("Customer Risk Segments")

    st.markdown("""
    Customers are grouped into four risk tiers based on
    order history, delay behavior, and purchasing patterns.
    """)

    segments_path = (
        "dataset/processed/customer_segments.csv"
    )

    if os.path.exists(segments_path):

        seg_df = pd.read_csv(segments_path)

        tier_counts = (
            seg_df["Risk Tier"]
            .value_counts()
        )

        cols = st.columns(4)

        for i, (tier, count) in enumerate(
            tier_counts.items()
        ):
            cols[i].metric(tier, count)

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(
            [
                "Segment Distribution",
                "Customer Visualization",
                "Risk Profiles"
            ]
        )

        with tab1:

            if os.path.exists(
                "reports/figures/cluster_distribution.png"
            ):
                st.image(
                    "reports/figures/cluster_distribution.png",
                    use_container_width=True
                )

        with tab2:

            if os.path.exists(
                "reports/figures/cluster_pca.png"
            ):
                st.image(
                    "reports/figures/cluster_pca.png",
                    use_container_width=True
                )

            st.markdown("""
            Principal Component Analysis (PCA) projects
            customer features into two dimensions for
            visualization. Clear separation suggests that
            the clustering model identified distinct
            customer groups.
            """)

        with tab3:

            profile = (
                seg_df.groupby("Risk Tier")
                .agg(
                    Customers=(
                        "Order Customer Id",
                        "count"
                    ),
                    Avg_Late_Rate=(
                        "late_rate",
                        "mean"
                    ),
                    Avg_Delay_Gap=(
                        "avg_delay_gap",
                        "mean"
                    ),
                    Avg_Order_Value=(
                        "avg_order_value",
                        "mean"
                    ),
                    Avg_Total_Orders=(
                        "total_orders",
                        "mean"
                    )
                )
                .round(3)
            )

            st.dataframe(profile)

            st.markdown("""
            Customers in higher-risk segments tend to exhibit
            higher late-delivery rates and larger delay gaps.
            These groups may benefit from proactive operational
            monitoring and communication.
            """)

    else:
        st.warning(
            "Run the clustering notebook to generate customer_segments.csv."
        )