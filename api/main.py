"""
RetailIQ — FastAPI Backend (v2.0 — SQL bugs fixed)
Fixes:
  1. /insights/segments → o.order_id (AmbiguousColumn fix)
  2. /insights/delivery  → CTE wrapper (alias in ORDER BY fix)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib, json, os, numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI(
    title       = "RetailIQ API",
    description = "E-Commerce Customer Intelligence",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"

try:
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    with open(MODEL_DIR / "feature_cols.json") as f:
        FEATURE_COLS = json.load(f)
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    OPTIMAL_THRESHOLD = metadata.get("optimal_threshold", 0.36)
    print(f"✅ Model loaded — AUC: {metadata.get('test_auc')} | Features: {len(FEATURE_COLS)}")
except Exception as e:
    model, FEATURE_COLS, metadata, OPTIMAL_THRESHOLD = None, [], {}, 0.36
    print(f"⚠️  Model load failed: {e}")

DB_URL = (
    f"postgresql://{os.getenv('DB_USER', 'retailiq_user')}"
    f":{os.getenv('DB_PASSWORD', 'retailiq123')}"
    f"@{os.getenv('DB_HOST', 'localhost')}"
    f":{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME', 'retailiq')}"
)
try:
    engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5)
    print("✅ Database connected")
except Exception as e:
    engine = None
    print(f"⚠️  DB connection failed: {e}")


# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════

class FirstOrderFeatures(BaseModel):
    order_value:            float = Field(..., ge=0)
    freight_value:          float = Field(..., ge=0)
    freight_pct:            float = Field(..., ge=0)
    item_count:             int   = Field(..., ge=1)
    avg_item_price:         float = Field(..., ge=0)
    actual_delivery_days:   float = Field(..., ge=0)
    promised_delivery_days: float = Field(..., ge=1)
    delay_hours:            float = Field(0.0, ge=0)
    delivered_on_time:      int   = Field(..., ge=0, le=1)
    delivery_speed_ratio:   float = Field(..., ge=0)
    review_score:           float = Field(..., ge=1, le=5)
    positive_review:        int   = Field(..., ge=0, le=1)
    bad_review:             int   = Field(..., ge=0, le=1)
    review_response_days:   float = Field(0.0, ge=0)
    payment_type:           str
    installments:           int   = Field(1, ge=1)
    payment_methods_used:   int   = Field(1, ge=1)
    category_grouped:       str
    product_weight_g:       float = Field(..., ge=0)
    product_photos_qty:     int   = Field(0, ge=0)
    purchase_month:         int   = Field(..., ge=1, le=12)
    purchase_dow:           int   = Field(..., ge=0, le=6)
    purchase_hour:          int   = Field(..., ge=0, le=23)
    state_grouped:          str

    class Config:
        json_schema_extra = {
            "example": {
                "order_value": 189.90, "freight_value": 18.50,
                "freight_pct": 9.74, "item_count": 2,
                "avg_item_price": 94.95,
                "actual_delivery_days": 8.0, "promised_delivery_days": 12.0,
                "delay_hours": 0.0, "delivered_on_time": 1,
                "delivery_speed_ratio": 0.67,
                "review_score": 5.0, "positive_review": 1, "bad_review": 0,
                "review_response_days": 1.0,
                "payment_type": "credit_card", "installments": 3,
                "payment_methods_used": 1,
                "category_grouped": "health_beauty",
                "product_weight_g": 400.0, "product_photos_qty": 4,
                "purchase_month": 11, "purchase_dow": 1, "purchase_hour": 19,
                "state_grouped": "SP",
            }
        }


class BatchRequest(BaseModel):
    customers: List[FirstOrderFeatures]


class ChurnResponse(BaseModel):
    churn_probability: float
    risk_label:        str
    risk_score:        int
    will_return:       bool
    recommendation:    str
    top_risk_factors:  List[str]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def get_risk_info(prob: float):
    if prob >= 0.75:
        return "HIGH",   False, "Immediate retention offer — discount or follow-up call"
    elif prob >= 0.50:
        return "MEDIUM", False, "Send personalised win-back email within 7 days"
    elif prob >= 0.25:
        return "LOW",    True,  "Monitor — light nurture sequence"
    else:
        return "SAFE",   True,  "Loyal customer — reward with loyalty programme"


def get_risk_factors(d: dict) -> List[str]:
    factors = []
    if d.get("bad_review") == 1:
        factors.append("Gave a 1–2 star review on first order")
    if d.get("delay_hours", 0) > 24:
        factors.append(f"Order delivered {d['delay_hours']:.0f}h late")
    if d.get("delivered_on_time") == 0:
        factors.append("First delivery arrived after promised date")
    if d.get("freight_pct", 0) > 30:
        factors.append(f"High freight cost ({d['freight_pct']:.1f}% of order value)")
    if d.get("review_score", 5) < 3:
        factors.append(f"Low satisfaction score ({d['review_score']:.1f}/5)")
    if d.get("installments", 1) >= 6:
        factors.append(f"High-installment purchase ({d['installments']}x) — expectations high")
    if d.get("actual_delivery_days", 0) > 20:
        factors.append(f"Very long delivery ({d['actual_delivery_days']:.0f} days)")
    if not factors:
        factors.append("No significant churn risk factors detected")
    return factors[:3]


def features_to_df(customer: FirstOrderFeatures) -> pd.DataFrame:
    d = customer.dict()
    df = pd.DataFrame([d])
    cat_cols = ['payment_type', 'category_grouped', 'state_grouped']
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    df_aligned = df_encoded.reindex(columns=FEATURE_COLS, fill_value=0)
    return df_aligned


def predict_one(customer: FirstOrderFeatures) -> ChurnResponse:
    df_input = features_to_df(customer)
    prob     = float(model.predict_proba(df_input)[0][1])
    label, will_return, recommendation = get_risk_info(prob)
    return ChurnResponse(
        churn_probability = round(prob, 4),
        risk_label        = label,
        risk_score        = int(prob * 100),
        will_return       = will_return,
        recommendation    = recommendation,
        top_risk_factors  = get_risk_factors(customer.dict()),
    )


def db_query(sql: str):
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return [dict(r._mapping) for r in rows]


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "service":   "RetailIQ API v2.0",
        "status":    "live",
        "model_auc": metadata.get("test_auc", "not loaded"),
        "features":  len(FEATURE_COLS),
        "docs":      "/docs",
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    if not metadata:
        raise HTTPException(503, "Model not loaded")
    return metadata


@app.post("/predict/churn", response_model=ChurnResponse, tags=["Predictions"])
def predict_churn(customer: FirstOrderFeatures):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return predict_one(customer)


@app.post("/predict/churn/batch", tags=["Predictions"])
def predict_churn_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if len(request.customers) > 500:
        raise HTTPException(400, "Max 500 customers per batch")
    results = [predict_one(c) for c in request.customers]
    probs   = [r.churn_probability for r in results]
    labels  = [r.risk_label for r in results]
    return {
        "total_customers": len(results),
        "summary": {
            "high_risk":             labels.count("HIGH"),
            "medium_risk":           labels.count("MEDIUM"),
            "low_risk":              labels.count("LOW"),
            "safe":                  labels.count("SAFE"),
            "avg_churn_probability": round(float(np.mean(probs)), 4),
            "pct_will_not_return":   round(
                sum(p > OPTIMAL_THRESHOLD for p in probs) / len(probs) * 100, 1
            ),
        },
        "predictions": [
            {
                "index":             i,
                "churn_probability": r.churn_probability,
                "risk_label":        r.risk_label,
                "risk_score":        r.risk_score,
                "will_return":       r.will_return,
                "recommendation":    r.recommendation,
            }
            for i, r in enumerate(results)
        ],
    }


@app.get("/insights/summary", tags=["Insights"])
def get_summary():
    rows = db_query("""
        SELECT
            COUNT(DISTINCT order_id)                    AS total_orders,
            COUNT(DISTINCT customer_id)                 AS total_customers,
            ROUND(SUM(order_revenue)::NUMERIC, 2)       AS total_revenue,
            ROUND(AVG(order_revenue)::NUMERIC, 2)       AS avg_order_value,
            ROUND(AVG(review_score)::NUMERIC, 2)        AS avg_rating,
            ROUND(AVG(CASE WHEN delivery_status = 'Late'
                           THEN 1.0 ELSE 0.0 END)*100
                  ::NUMERIC, 2)                         AS late_delivery_pct
        FROM analytics.orders_enriched
        WHERE order_status = 'delivered'
    """)
    return rows[0] if rows else {}


@app.get("/insights/revenue", tags=["Insights"])
def get_revenue():
    return db_query("""
        SELECT
            TO_CHAR(month, 'Mon YYYY')  AS month_label,
            month::TEXT                 AS month_iso,
            orders,
            customers,
            revenue,
            aov,
            avg_rating,
            late_pct
        FROM analytics.monthly_kpis
        WHERE month >= '2017-01-01'
        ORDER BY month
    """)


@app.get("/insights/segments", tags=["Insights"])
def get_segments():
    # FIX: o.order_id fully qualified — removes AmbiguousColumn error
    return db_query("""
        WITH rfm AS (
            SELECT
                o.customer_id,
                NTILE(5) OVER (ORDER BY MAX(o.order_purchase_ts) DESC) AS r,
                NTILE(5) OVER (ORDER BY COUNT(DISTINCT o.order_id))    AS f,
                NTILE(5) OVER (ORDER BY SUM(oi.price))                 AS m
            FROM raw.orders o
            JOIN raw.order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY o.customer_id
        )
        SELECT
            CASE
                WHEN (r+f+m) >= 13                THEN 'Champions'
                WHEN (r+f+m) >= 10 AND f >= 3     THEN 'Loyal Customers'
                WHEN r >= 4 AND f <= 2             THEN 'Promising'
                WHEN r <= 2 AND (f+m) >= 6        THEN 'At Risk'
                WHEN (r+f+m) <= 5                 THEN 'Lost'
                ELSE                                   'Needs Attention'
            END                                           AS segment,
            COUNT(*)                                      AS customer_count,
            ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2) AS pct
        FROM rfm
        GROUP BY 1
        ORDER BY customer_count DESC
    """)


@app.get("/insights/delivery", tags=["Insights"])
def get_delivery():
    # FIX: CTE wrapper so ORDER BY can reference the alias delivery_bucket
    return db_query("""
        WITH classified AS (
            SELECT
                CASE
                    WHEN o.order_delivered_ts IS NULL
                        THEN 'Not Delivered'
                    WHEN o.order_delivered_ts <= o.order_estimated_ts
                        THEN 'On Time'
                    WHEN o.order_delivered_ts <= o.order_estimated_ts + INTERVAL '1 day'
                        THEN 'Late: 0-1d'
                    WHEN o.order_delivered_ts <= o.order_estimated_ts + INTERVAL '3 days'
                        THEN 'Late: 1-3d'
                    WHEN o.order_delivered_ts <= o.order_estimated_ts + INTERVAL '7 days'
                        THEN 'Late: 3-7d'
                    ELSE 'Late: 7d+'
                END              AS delivery_bucket,
                r.review_score
            FROM raw.orders o
            JOIN raw.reviews r ON o.order_id = r.order_id
            WHERE o.order_status IN ('delivered', 'shipped')
              AND r.review_score IS NOT NULL
        )
        SELECT
            delivery_bucket,
            COUNT(*)                                              AS order_count,
            ROUND(AVG(review_score)::NUMERIC, 2)                 AS avg_rating,
            ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2)        AS pct_of_orders
        FROM classified
        GROUP BY delivery_bucket
        ORDER BY
            CASE delivery_bucket
                WHEN 'On Time'    THEN 1
                WHEN 'Late: 0-1d' THEN 2
                WHEN 'Late: 1-3d' THEN 3
                WHEN 'Late: 3-7d' THEN 4
                WHEN 'Late: 7d+'  THEN 5
                ELSE 6
            END
    """)


@app.get("/insights/top-categories", tags=["Insights"])
def get_top_categories():
    return db_query("""
        SELECT
            COALESCE(ct.product_category_name_english,
                     p.product_category_name, 'Unknown')  AS category,
            COUNT(DISTINCT o.order_id)                    AS orders,
            ROUND(SUM(oi.price)::NUMERIC, 2)              AS revenue,
            ROUND(AVG(r.review_score)::NUMERIC, 2)        AS avg_rating
        FROM raw.order_items oi
        JOIN raw.orders               o  ON oi.order_id   = o.order_id
        JOIN raw.products             p  ON oi.product_id = p.product_id
        LEFT JOIN raw.reviews         r  ON o.order_id    = r.order_id
        LEFT JOIN raw.category_translation ct
            ON p.product_category_name = ct.product_category_name
        WHERE o.order_status = 'delivered'
        GROUP BY 1
        HAVING COUNT(DISTINCT o.order_id) >= 100
        ORDER BY revenue DESC
        LIMIT 15
    """)