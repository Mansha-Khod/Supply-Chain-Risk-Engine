from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_overview():
    return {
        "title": "Supply Chain Risk Engine",
        "description": (
            "An end-to-end machine learning system that predicts shipment "
            "delays, estimates delay severity, and segments customers by "
            "risk using more than 180,000 real logistics records."
        ),
        "metrics": {
            "classification": {"model": "XGBoost", "metric": "AUC-ROC", "score": "0.93"},
            "regression": {"model": "Random Forest", "metric": "MAE", "score": "0.48 Days"},
            "segmentation": {"model": "K-Means", "metric": "Risk Tiers", "score": "4"},
        },
        "pipeline": [
            {"stage": "New Shipment Order", "description": "Input data"},
            {"stage": "Stage 1: Classification", "description": "Will the shipment be delayed? (XGBoost)"},
            {"stage": "Stage 2: Regression", "description": "Estimate delay duration in days (Random Forest)"},
            {"stage": "Stage 3: Customer Profiling", "description": "Assign customer risk tier (K-Means)"},
            {"stage": "Model Explanation", "description": "Explain prediction drivers using SHAP"},
        ],
        "insights": [
            "First Class shipping shows the highest late-delivery rate.",
            "Customer history is the strongest predictor of future delays.",
            "The regression model explains a meaningful share of delivery variance.",
            "A small group of customers contributes disproportionately to delayed shipments.",
        ],
    }
