import io
import pandas as pd
from fastapi import APIRouter,File,HTTPException,UploadFile
from fastapi.responses import StreamingResponse

from app.model_loader import get_models

router=APIRouter()

def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    raise HTTPException(status_code=400, detail="Could not parse CSV: unsupported encoding.")

def _score(data:pd.DataFrame)-> pd.DataFrame:
     models=get_models()
     feature_names=models.xgb_feature_names

     missing=[f for f in feature_names if f not in data.columns]
     if missing:
         raise HTTPException(
             status_code=422,
             detail={"message":"Missing Required Columns","missing_columns":missing},
         )
     X=data[feature_names]
     predictions=models.xgb_model.predict(X)
     probabilities=models.xgb_model.predict_proba(X)[:,1]

     result=data.copy()
     result['Delay Risk']=predictions
     result['Risk Probability']=probabilities
     result['Risk Label']=result['Risk Probability'].apply(
         lambda x:"High Risk" if x>70 else ("Medium Risk" if x>40 else "Low Risk")

     )
     ordered_cols=['Risk Label',"Risk Probability"]+[
         c for c in result.columns if c not in ("Delay Risk","Risk Probability","Risk Label")

     ]
     return result[ordered_cols]

@router.get("/features")
def get_required_features():
    models=get_models()
    return {"required_columns":models.xgb_feature_names}

@router.post("")
async def predict_risk(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400,detail='Please upload a .csv file.')

    contents= await file.read()
    data=_read_csv(contents)
    results=_score(data)

    return{
        "total_shipments":len(results),
        'high_risk_count':int((results['Risk Label']=='High Risk').sum()),
        'medium_risk_count':int((results['RiskLabel']=='Medium Risk').sum()),
        'low_risk_count':int((results['RiskLabel']=='Low Risk').sum()),
        'average_risk_percent':round(float(results['Risk Probabilities'].mean()),1),
        "preview":results.head(50).to_dict(orient='records'),

    }

@router.post("/download")
async def predict_and_download(file:UploadFile=File(...)):
    if file.filename.lower().endswith('csv'):
        raise HTTPException(status_code=400,detail="Please Upload a .csv file")

    contents=await file.read()
    data=_read_csv(contents)
    results=_score(data)

    buf=io.StringIO()
    results.to_csv(buf,index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={'Content-Disposition':"attachment; filename=shipment_predictions.csv"},
    )
    
