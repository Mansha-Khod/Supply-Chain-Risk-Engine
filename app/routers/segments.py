import pandas as pd
from fastapi import APIRouter,HTTPException
from app.config import CUSTOMER_SEGMENTS_CSV,FIGURES_DIR

router=APIRouter()

@router.get("")
def get_segments():
    if not CUSTOMER_SEGMENTS_CSV.exists():
        raise HTTPException(status_code=404,detail="customer_segments.csv not found")

    seg_df=pd.read_csv(CUSTOMER_SEGMENTS_CSV)
    tier_counts=seg_df['Risk Tier'].value_counts().to_dict()

    profile=(
        seg_df.groupby("Risk Tier")
        .agg(
            Customers=("Order Customer Id",'count'),
            Avg_Late_Rate=("late_rate","mean"),
            Avg_Delay_Gap=("avg_delay_gap",'mean'),
            Avg_Order_Value=("avg_order_value",'mean'),
            Avg_Total_Orders=("total_orders",'mean'),
        ).round(3)
     .reset_index()
     .to_dict(orient="records")
    )
    return {
        "tier_counts": tier_counts,
        "profile": profile,
        "distribution_image_url": "/assets/cluster_distribution.png"
        if (FIGURES_DIR / "cluster_distribution.png").exists() else None,
        "pca_image_url": "/assets/cluster_pca.png"
        if (FIGURES_DIR / "cluster_pca.png").exists() else None,

    }