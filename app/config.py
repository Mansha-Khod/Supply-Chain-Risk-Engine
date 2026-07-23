from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent

MODELS_DIR=BASE_DIR/"modles"
DATASET_DIR=BASE_DIR/'dataset'
PROCESSED_DIR=DATASET_DIR/'processed'
FIGURES_DIR=BASE_DIR/'reports'/'figures'

XGB_MODEL=MODELS_DIR/'xgb_classifier.pkl'
RF_MODEL=MODELS_DIR/'rf_regressor.pkl'
SCALER_PATH=MODELS_DIR/'scaler.pkl'
SHAP_EXPLAINER_PATH=MODELS_DIR/'shap_explainer.pkl'

CUSTOMER_SEGMENTS_CSV=PROCESSED_DIR/'CUSTOMER_SEGMENTS.CSV'


