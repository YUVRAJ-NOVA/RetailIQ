<div align="center">

<img src="https://img.shields.io/badge/RetailIQ-Customer%20Intelligence%20Platform-4f8ef7?style=for-the-badge&logoColor=white" alt="RetailIQ"/>

# RetailIQ — E-Commerce Customer Intelligence Platform

**End-to-end data platform built on 100K+ real orders.**  
From raw CSV to live ML predictions — PostgreSQL · XGBoost · FastAPI · Dashboard.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.777-ff6600?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8b5cf6?style=flat-square)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-34d399?style=flat-square)](LICENSE)

<br/>

[**Live Dashboard**](https://retailiq.pages.dev) · [**API Docs**](https://retailiq-api.onrender.com/docs) · [**The Story**](https://retailiq.pages.dev/story.html)

</div>

---

## What This Is

Most e-commerce companies know their revenue numbers. They don't know **why 93% of their customers never come back.**

RetailIQ answers that — with a complete data pipeline that goes from 9 raw CSV files to a live churn prediction API, backed by real findings from 99,441 real orders.

This is not a tutorial follow-along. Every table was designed, every SQL query was written to answer a real business question, and the ML model was rebuilt from scratch after I caught target leakage in v1.

---

## Key Results

| Metric | Value |
|---|---|
| Churn Model ROC-AUC | **0.777** |
| CV Stability (5-fold) | **0.760 ± 0.009** |
| Train-CV Gap | **0.016** — no overfitting |
| #1 Churn Predictor (SHAP) | **freight_value** |
| Customers Segmented | **96,478** via RFM |
| Revenue Analysed | **R$16.7M+** across 24 months |
| Late Delivery Rating Drop | **4.29 → 1.73** (7+ day delay) |

---

## Architecture

```text
  9 Raw CSVs (Kaggle Olist)
            │
            ▼
┌──────────────────────┐
│   Python Ingestion   │  Batch ingestion · SQLAlchemy · logging · error handling
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│    PostgreSQL 16     │  2 schemas (raw + analytics) · 9 tables · 7 indexes
│    Medallion Arch    │  Medallion architecture: raw → analytics layer
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│  SQL Analysis Layer  │  5 production queries · CTEs · window functions
│                      │  RFM segmentation · NTILE · PERCENTILE_CONT
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│     Feature Eng.     │  62 leakage-free features · first-order frame
│       + EDA          │  Pandas · Matplotlib · SHAP
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│    XGBoost Model     │  RandomizedSearchCV (30 iter) · 5-fold stratified CV
│       + SHAP         │  Leakage detected + fixed · model rebuilt from scratch
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│       FastAPI        │  9 endpoints · Pydantic validation · live DB + model
│       REST API       │  /predict/churn · /predict/churn/batch · /insights/*
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│      Dashboard       │  Pure HTML/JS · SVG charts · scroll-driven story page
│     + Story Page     │  Calls live API · responsive · dark theme
└──────────────────────┘
```
---

---

## Role Breakdown

<details>
<summary><b>🏗️ Data Engineer</b></summary>

- Designed 9-table relational schema in PostgreSQL with proper FK constraints and performance indexes
- Implemented medallion architecture — `raw` schema for ingestion, `analytics` schema for business views
- Built automated Python ingestion pipeline with batch inserts, logging, and error handling
- Created 3 production analytics views consumed by both FastAPI and dashboard
- Deployed FastAPI with 9 REST endpoints, CORS, Pydantic validation, and SQLAlchemy connection pooling

**Tech:** PostgreSQL 16 · SQLAlchemy · psycopg2 · Python · FastAPI · Uvicorn · python-dotenv

</details>

<details>
<summary><b>📊 Data Analyst</b></summary>

- Wrote 5 production-grade SQL queries answering real business questions
- **Query 01** — Revenue trend: `DATE_TRUNC`, `LAG` window function, MoM growth, cumulative revenue
- **Query 02** — RFM segmentation: 4-CTE chain, `NTILE(5)` scoring, business action mapping across 96,478 customers
- **Query 03** — Delivery vs satisfaction: `PERCENTILE_CONT`, `INTERVAL` arithmetic, rating heatmap
- **Query 04** — Product health scoring: `RANK()`, composite KPI, strategic classification
- **Query 05** — Seller scorecard: `DENSE_RANK`, `PERCENT_RANK`, composite performance score
- Key finding: late delivery (7+ days) drops avg rating from **4.29 → 1.73** — a 60% satisfaction collapse

**Tech:** PostgreSQL · CTEs · Window Functions · NTILE · PERCENTILE_CONT · FILTER aggregation

</details>

<details>
<summary><b>🤖 Data Scientist</b></summary>

- Detected and fixed **target leakage** in v1 model (99.9% precision → CV AUC 0.56 → correctly identified as memorisation)
- Redesigned ML frame: first-order signals only — recency and frequency completely excluded as features
- Final model: XGBoost · `RandomizedSearchCV` (30 iterations) · 5-fold stratified CV · `scale_pos_weight` for class imbalance
- **ROC-AUC: 0.777** · CV: 0.760 ± 0.009 · Train-CV gap: 0.016 (stable)
- SHAP explainability on 2,000 test samples — `freight_value` identified as #1 churn driver
- Business insight: high freight-to-order-value ratio = "surprise cost shock" = silent churn

**Tech:** XGBoost · SHAP · Scikit-learn · Pandas · NumPy · Joblib · Matplotlib

</details>

<details>
<summary><b>⚡ ML Engineer / AI Engineer</b></summary>

- Serialised trained XGBoost model as `.pkl` with feature column manifest (`feature_cols.json`) for reproducible inference
- Built real-time prediction endpoint: `POST /predict/churn` — Pydantic-validated input, one-hot encoding aligned to training schema, churn probability + risk label + SHAP-based risk factors returned
- Batch prediction endpoint: `POST /predict/churn/batch` — up to 500 customers, aggregate summary + per-customer scores
- Model metadata stored as JSON — AUC, CV scores, optimal threshold, top SHAP features — queryable via `/model/info`
- Production-safe: no feature leakage in inference path, input validation catches schema mismatches

**Tech:** FastAPI · Pydantic · Joblib · XGBoost inference · SHAP · REST API design

</details>

<details>
<summary><b>📈 Business Intelligence / Analytics Engineer</b></summary>

- 3 analytics views designed for downstream consumption — `orders_enriched`, `monthly_kpis`, `customer_ltv`
- Views power both the FastAPI layer and the dashboard with zero duplication
- Dashboard built as pure HTML/JS — 5 analysis panels + scroll-driven story page
- All charts are custom SVG — not Chart.js defaults — for precise business communication
- Panels: Executive Summary · Revenue Intelligence · Customer Segments · Delivery Impact · Churn Predictor

**Tech:** PostgreSQL Views · HTML · CSS · Vanilla JS · SVG · REST API consumption

</details>

---

## SQL Highlights

```sql
-- RFM Segmentation — 4-CTE chain, NTILE scoring, business action mapping
WITH rfm_raw AS (
    SELECT customer_id,
        EXTRACT(DAY FROM (ref_date - MAX(order_purchase_ts)))::INT AS recency_days,
        COUNT(DISTINCT order_id)                                    AS frequency,
        SUM(price)                                                  AS monetary
    FROM raw.orders o
    JOIN raw.order_items oi ON o.order_id = oi.order_id
    CROSS JOIN (SELECT MAX(order_purchase_ts) AS ref_date FROM raw.orders) r
    WHERE order_status = 'delivered'
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency)         AS f_score,
        NTILE(5) OVER (ORDER BY monetary)          AS m_score
    FROM rfm_raw
)
SELECT
    CASE
        WHEN (r_score + f_score + m_score) >= 13 THEN 'Champions'
        WHEN (r_score + f_score + m_score) >= 10
             AND f_score >= 3                    THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2       THEN 'Promising'
        WHEN r_score <= 2 AND (f_score+m_score) >= 6 THEN 'At Risk'
        WHEN (r_score + f_score + m_score) <= 5  THEN 'Lost'
        ELSE                                          'Needs Attention'
    END                                              AS segment,
    COUNT(*)                                         AS customers,
    ROUND(AVG(monetary)::NUMERIC, 2)                 AS avg_ltv
FROM rfm_scored
GROUP BY 1
ORDER BY avg_ltv DESC;
```

---

## ML Model — Leakage Detection & Fix

**v1 (Discarded)** — Classic target leakage:
```text
recency_days > 180  =  churned label

recency_days        =  model feature  ← same variable, model memorised rule

Result: 99.9% train precision · CV AUC 0.56 · meaningless
```
**v2 (Production)** — Leakage-free first-order frame:
* **Label:** Did customer ever place a 2nd order? (binary)
* **Features:** First-order signals only — delivery speed, freight cost, review score, payment pattern, product category
* **Excluded:** `recency_days`, `frequency` — both derived from label source
* **Result:** ROC-AUC 0.777 · CV 0.760 ± 0.009 · SHAP meaningful

**Top SHAP predictors (from production model):**
```text
freight_value      ████████████████████  0.196  ← #1 churn driver
installments       █████████████████     0.171
avg_item_price     █████████████         0.150
promised_days      █████████████         0.134
actual_days        ██████████            0.108
review_score       █████████             0.094
```
---

## Project Structure

```text
RetailIQ/
│
├── 📁 data/
│   ├── raw/                     ← Original CSVs (gitignored)
│   └── sample/                  ← 500-row sample + churn features
│
├── 📁 database/
│   ├── schema.sql               ← 9-table schema + indexes
│   └── analytics_views.sql      ← 3 production views
│
├── 📁 ingestion/
│   └── load_data.py             ← CSV → PostgreSQL pipeline
│
├── 📁 analysis/
│   ├── sql/
│   │   ├── 01_revenue_analysis.sql
│   │   ├── 02_rfm_segmentation.sql
│   │   ├── 03_delivery_performance.sql
│   │   ├── 04_product_analysis.sql
│   │   └── 05_seller_performance.sql
│   └── notebooks/
│       ├── 01_EDA.ipynb         ← 15+ visualisations · dark theme
│       └── 02_churn_model.ipynb ← Leakage fix · XGBoost · SHAP
│
├── 📁 ml/
│   └── models/
│       ├── churn_model.pkl      ← Trained XGBoost
│       ├── shap_explainer.pkl   ← TreeExplainer
│       ├── feature_cols.json    ← 62 feature names (inference schema)
│       └── model_metadata.json  ← AUC · CV · threshold · top features
│
├── 📁 api/
│   └── main.py                  ← FastAPI · 9 endpoints · Pydantic
│
├── 📁 dashboard/
│   ├── index.html               ← 5-panel analytics dashboard
│   └── story.html               ← Scroll-driven project narrative
│
├── 📁 reports/
│   ├── findings.md              ← Key business insights
│   └── *.png                    ← EDA + model evaluation charts
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + model AUC |
| `GET` | `/model/info` | Full model metadata |
| `POST` | `/predict/churn` | Single customer churn prediction |
| `POST` | `/predict/churn/batch` | Batch prediction (up to 500) |
| `GET` | `/insights/summary` | Live KPI summary from DB |
| `GET` | `/insights/revenue` | Monthly revenue trend |
| `GET` | `/insights/segments` | RFM segment distribution |
| `GET` | `/insights/delivery` | Delivery performance vs rating |
| `GET` | `/insights/top-categories` | Top 15 categories by revenue |

**Sample prediction request:**
```bash
curl -X POST https://retailiq-api.onrender.com/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "order_value": 189.90,
    "freight_value": 18.50,
    "freight_pct": 9.74,
    "actual_delivery_days": 8,
    "promised_delivery_days": 12,
    "delay_hours": 0,
    "delivered_on_time": 1,
    "delivery_speed_ratio": 0.67,
    "review_score": 5,
    "positive_review": 1,
    "bad_review": 0,
    "payment_type": "credit_card",
    "installments": 3,
    "category_grouped": "health_beauty",
    "product_weight_g": 400,
    "purchase_month": 11,
    "purchase_dow": 1,
    "purchase_hour": 19,
    "state_grouped": "SP",
    "item_count": 2,
    "avg_item_price": 94.95,
    "freight_response_days": 0,
    "payment_methods_used": 1,
    "product_photos_qty": 4
  }'
```

**Response:**
```json
{
  "churn_probability": 0.1247,
  "risk_label": "SAFE",
  "risk_score": 12,
  "will_return": true,
  "recommendation": "Loyal customer — reward with loyalty programme",
  "top_risk_factors": ["No significant churn risk factors detected"]
}
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YUVRAJ-NOVA/RetailIQ.git
cd RetailIQ

# 2. Environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# 4. Database
psql -U retailiq_user -d retailiq -f database/schema.sql
python ingestion/load_data.py

# 5. API
cd api && uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs

# 6. Dashboard
# Open dashboard/index.html in browser
```

**Dataset:** [Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — download and place CSVs in `data/raw/`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 · pgAdmin 4 |
| Ingestion | Python · SQLAlchemy · psycopg2 |
| Analysis | SQL · CTEs · Window Functions |
| EDA | Pandas · Matplotlib · Seaborn |
| ML | XGBoost · Scikit-learn · SHAP · Joblib |
| API | FastAPI · Pydantic · Uvicorn |
| Dashboard | HTML · CSS · Vanilla JS · SVG |
| Deployment | Render · Neon · Cloudflare Pages |

---

## Four Business Findings

> **01 — Delivery SLA is the #1 retention lever**  
> Late delivery (7+ days) drops avg rating from 4.29 → 1.73. Even 1 day late: 4.29 → 2.6. No discount needed — fix logistics.

> **02 — Freight cost predicts churn better than satisfaction score**  
> SHAP #1 predictor. Surprise shipping cost = silent churn. Free shipping above R$150 would directly address this.

> **03 — Champions have 10× the LTV of Lost customers**  
> R$318 avg LTV vs R$32. 15,526 Champions identified. Referral programme targeting this segment compounds growth.

> **04 — health_beauty is the benchmark category**  
> R$1.24M revenue · high satisfaction · manageable logistics. Every other category should study it.

---

<div align="center">

**Built by [Yuvraj](https://github.com/YUVRAJ-NOVA)**  


</div>