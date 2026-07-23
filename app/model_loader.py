from functools import lru_cache
from app.config import XGB_MODEL_PATH,RF_MODEL_PATH,SCALER_PATH,SHAP_EXPLAINER_PATH
import joblib

class Models:
    def __init__(self):
        self.xgb_model=joblib.load(XGB_MODEL_PATH)
        self.rf_model=joblib.load(RF_MODEL_PATH)
        self.scaler=joblib.load(SCALER_PATH)
        self.explainer=joblib.load(SHAP_EXPLAINER_PATH)
        self.xgb_feature_names=self.xgb_model.get_booster().feature_names


@lru_cache(maxsize=1)
def get_models()->Models:
    return Models()       