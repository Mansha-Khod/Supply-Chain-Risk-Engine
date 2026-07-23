from app.config import FIGURES_DIR
from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_explainability():
    return {
        "summary": {
            "title": "Global Feature Importance",
            "description": "Identify which features influence delay predictions across all shipments.",
            "image_url": "/assets/shap_summary.png",
            "available": (FIGURES_DIR / "shap_summary.png").exists(),
            "interpretation": [
                "Features are ranked by overall importance.",
                "Each point represents a shipment.",
                "Higher feature values are shown in red, lower in blue.",
                "Points on the right increase delay risk; on the left, decrease it.",
            ],
        },
        "waterfall": {
            "title": "Individual Prediction Explanation",
            "description": "Understand why a specific shipment was classified as high risk.",
            "image_url": "/assets/shap_waterfall.png",
            "available": (FIGURES_DIR / "shap_waterfall.png").exists(),
            "interpretation": [
                "The base value is the average model prediction.",
                "Each bar represents a feature's contribution.",
                "Positive contributions increase delay risk; negative decrease it.",
            ],
        },
    }