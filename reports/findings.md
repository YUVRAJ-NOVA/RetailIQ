# RetailIQ — Key Business Findings

## Dataset
- **Source**: Olist Brazilian E-Commerce (Kaggle)
- **Scale**: 99,441 orders | 96,478 unique customers | 9 interconnected tables
- **Period**: September 2016 – October 2018

---

## Finding 1 — Delivery is the Single Biggest Retention Driver
Late delivery drops average review rating from **4.29 (on time) → 1.73 (7+ days late)** — a 60% satisfaction collapse. Even a **1-day delay** sends rating from 4.29 to 2.6.

**Business action**: Tighten SLA on logistics partners, especially for heavy/bulky categories where delay risk is highest.

---

## Finding 2 — 93% of Customers Never Return
RFM analysis shows only ~7% of customers placed more than one order. Single-purchase churn is the #1 revenue leak. Champions (top 16%) generate **10x the LTV** of Lost customers.

**Business action**: Invest in first-order experience — the first delivery interaction determines whether a customer ever returns.

---

## Finding 3 — Freight Cost Kills Repeat Purchase
SHAP analysis on the churn model (AUC 0.777) identified **freight_value** as the #1 churn predictor. High freight relative to order value signals "surprise cost" shock — customers feel deceived and don't return.

**Business action**: Free shipping threshold programme for orders above R$150 would directly address this.

---

## Finding 4 — health_beauty is the Star Category
Revenue rank #1 with high satisfaction ratings. Heavy/bulky categories (bed_bath_table, furniture_decor) show high revenue but elevated churn risk due to complex logistics.

**Business action**: Category-specific SLA policies — premium logistics for heavy items.

---

## Churn Model Summary
| Metric | Value |
|---|---|
| Algorithm | XGBoost |
| ROC-AUC (test) | 0.777 |
| CV AUC (5-fold) | 0.760 ± 0.009 |
| Train AUC | 0.869 |
| Optimal Threshold | 0.36 |
| Features | 40+ (first-order signals only) |
| Leakage check | PASSED — recency & frequency excluded |
| Top predictor | freight_value |