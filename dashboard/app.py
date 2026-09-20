# dashboard/app.py
"""
RetailIQ — Plotly Dash Dashboard
5 Panels:
  1. Executive Summary   → KPI cards (live from DB)
  2. Revenue Intelligence → Monthly trend + growth
  3. Customer Segments   → RFM donut + LTV bars
  4. Delivery Impact     → Rating vs delivery bucket
  5. Churn Predictor     → Live prediction form
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Config 
API_BASE = "http://localhost:8000"   # FastAPI URL

# Colors — same dark theme as notebooks
COLORS = {
    "bg":         "#0d1117",
    "card":       "#161b22",
    "border":     "#30363d",
    "text":       "#c9d1d9",
    "muted":      "#8b949e",
    "primary":    "#58a6ff",
    "success":    "#3fb950",
    "warning":    "#d29922",
    "danger":     "#f85149",
    "purple":     "#bc8cff",
    "orange":     "#ffa657",
}

CHART_LAYOUT = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["card"],
    font          = dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin        = dict(l=40, r=20, t=40, b=40),
    xaxis         = dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
    yaxis         = dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
)

# App 
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap",
    ],
    title="RetailIQ Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server   # Expose for Gunicorn (Render deployment later)


# HELPER: Fetch from API

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"API error {endpoint}: {e}")
        return None



# COMPONENT BUILDERS


def kpi_card(title: str, value: str, subtitle: str = "", color: str = "primary"):
    color_map = {
        "primary": COLORS["primary"],
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "danger":  COLORS["danger"],
        "purple":  COLORS["purple"],
        "orange":  COLORS["orange"],
    }
    border_color = color_map.get(color, COLORS["primary"])
    return dbc.Card([
        dbc.CardBody([
            html.P(title, style={
                "color": COLORS["muted"],
                "fontSize": "0.75rem",
                "textTransform": "uppercase",
                "letterSpacing": "1px",
                "marginBottom": "4px",
                "fontWeight": "600",
            }),
            html.H3(value, style={
                "color": border_color,
                "fontWeight": "700",
                "fontSize": "1.8rem",
                "margin": "0",
            }),
            html.P(subtitle, style={
                "color": COLORS["muted"],
                "fontSize": "0.78rem",
                "marginTop": "4px",
                "marginBottom": "0",
            }) if subtitle else html.Div(),
        ])
    ], style={
        "backgroundColor":  COLORS["card"],
        "border":           f"1px solid {COLORS['border']}",
        "borderLeft":       f"3px solid {border_color}",
        "borderRadius":     "8px",
        "height":           "100%",
    })


def section_header(title: str, subtitle: str = ""):
    return html.Div([
        html.H4(title, style={
            "color": COLORS["text"],
            "fontWeight": "700",
            "marginBottom": "4px",
        }),
        html.P(subtitle, style={
            "color":        COLORS["muted"],
            "fontSize":     "0.85rem",
            "marginBottom": "20px",
        }) if subtitle else html.Div(),
    ])



# LAYOUT


SIDEBAR = html.Div([
    html.Div([
        html.H5("RetailIQ", style={
            "color":      COLORS["primary"],
            "fontWeight": "700",
            "fontSize":   "1.2rem",
        }),
        html.P("Customer Intelligence", style={
            "color":       COLORS["muted"],
            "fontSize":    "0.75rem",
            "marginBottom": "0",
        }),
    ], style={"padding": "20px 16px 16px"}),

    html.Hr(style={"borderColor": COLORS["border"], "margin": "0"}),

    dbc.Nav([
        dbc.NavLink([html.Span("📊", style={"marginRight": "8px"}), "Executive Summary"],
                    href="/", active="exact"),
        dbc.NavLink([html.Span("💰", style={"marginRight": "8px"}), "Revenue Intelligence"],
                    href="/revenue", active="exact"),
        dbc.NavLink([html.Span("👥", style={"marginRight": "8px"}), "Customer Segments"],
                    href="/segments", active="exact"),
        dbc.NavLink([html.Span("🚚", style={"marginRight": "8px"}), "Delivery Impact"],
                    href="/delivery", active="exact"),
        dbc.NavLink([html.Span("🔮", style={"marginRight": "8px"}), "Churn Predictor"],
                    href="/predict", active="exact"),
    ], vertical=True, pills=True, style={"padding": "12px 8px"}),

    html.Div([
        html.Hr(style={"borderColor": COLORS["border"]}),
        html.P([
            html.Span("Model AUC: ", style={"color": COLORS["muted"]}),
            html.Span("0.777", style={"color": COLORS["success"], "fontWeight": "600"}),
        ], style={"fontSize": "0.78rem", "padding": "0 16px"}),
        html.P([
            html.Span("Dataset: ", style={"color": COLORS["muted"]}),
            html.Span("100K+ Orders", style={"color": COLORS["text"]}),
        ], style={"fontSize": "0.78rem", "padding": "0 16px"}),
        html.P(
            f"Updated: {datetime.now().strftime('%d %b %Y')}",
            style={"color": COLORS["muted"], "fontSize": "0.72rem", "padding": "0 16px"},
        ),
    ], style={"marginTop": "auto", "paddingBottom": "16px"}),

], style={
    "width":           "220px",
    "minHeight":       "100vh",
    "backgroundColor": COLORS["card"],
    "borderRight":     f"1px solid {COLORS['border']}",
    "display":         "flex",
    "flexDirection":   "column",
    "position":        "fixed",
    "top":             "0",
    "left":            "0",
})

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    SIDEBAR,
    html.Div([
        html.Div(id="page-content", style={"padding": "28px"}),
    ], style={"marginLeft": "220px", "minHeight": "100vh",
              "backgroundColor": COLORS["bg"]}),
], style={"fontFamily": "Inter, sans-serif"})



# PAGE 1 — EXECUTIVE SUMMARY


def page_summary():
    data = api_get("/insights/summary") or {}
    revenue  = data.get("total_revenue", 0)
    orders   = data.get("total_orders", 0)
    customers= data.get("total_customers", 0)
    aov      = data.get("avg_order_value", 0)
    rating   = data.get("avg_rating", 0)
    late_pct = data.get("late_delivery_pct", 0)

    return html.Div([
        section_header(
            "Executive Summary",
            "Live KPIs from 100K+ real Brazilian e-commerce orders (2016–2018)"
        ),

        dbc.Row([
            dbc.Col(kpi_card("Total Revenue", f"R${revenue:,.0f}",
                             "Delivered orders only", "success"), md=2),
            dbc.Col(kpi_card("Total Orders", f"{orders:,}",
                             "Delivered status", "primary"), md=2),
            dbc.Col(kpi_card("Unique Customers", f"{customers:,}",
                             "By customer_id", "purple"), md=2),
            dbc.Col(kpi_card("Avg Order Value", f"R${aov:.2f}",
                             "Per delivered order", "orange"), md=2),
            dbc.Col(kpi_card("Avg Rating", f"⭐ {rating:.2f}/5",
                             "Customer satisfaction", "warning"), md=2),
            dbc.Col(kpi_card("Late Deliveries", f"{late_pct:.1f}%",
                             "After promised date", "danger"), md=2),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Key Business Findings",
                                style={"color": COLORS["primary"], "fontWeight": "700"}),
                        html.Ul([
                            html.Li("Late delivery (even 1 day) drops avg rating from 4.3 → 2.6",
                                    style={"marginBottom": "8px"}),
                            html.Li("health_beauty is #1 revenue category — ⭐ star performer",
                                    style={"marginBottom": "8px"}),
                            html.Li("RFM: Champions have 10x LTV vs Lost customers",
                                    style={"marginBottom": "8px"}),
                            html.Li("Churn model (AUC 0.777): Freight cost + delivery speed are top predictors",
                                    style={"marginBottom": "8px"}),
                            html.Li("93% of customers are one-time buyers — retention is the #1 growth lever"),
                        ], style={"color": COLORS["text"], "fontSize": "0.88rem",
                                  "paddingLeft": "18px"}),
                    ])
                ], style={"backgroundColor": COLORS["card"],
                          "border": f"1px solid {COLORS['border']}",
                          "borderRadius": "8px"}),
            ], md=7),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Model Performance",
                                style={"color": COLORS["purple"], "fontWeight": "700"}),
                        _model_perf_table(),
                    ])
                ], style={"backgroundColor": COLORS["card"],
                          "border": f"1px solid {COLORS['border']}",
                          "borderRadius": "8px"}),
            ], md=5),
        ], className="g-3"),
    ])


def _model_perf_table():
    rows = [
        ("ROC-AUC",       "0.777",  COLORS["success"]),
        ("CV AUC (5-fold)","0.760 ± 0.009", COLORS["success"]),
        ("Train AUC",     "0.869",  COLORS["warning"]),
        ("Avg Precision", "0.978",  COLORS["muted"]),
        ("Optimal Threshold", "0.36", COLORS["primary"]),
        ("Training rows", "61,916", COLORS["text"]),
    ]
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(label, style={"color": COLORS["muted"],
                                      "fontSize": "0.82rem",
                                      "paddingRight": "16px",
                                      "paddingBottom": "6px"}),
                html.Td(val,   style={"color": color,
                                      "fontWeight": "600",
                                      "fontSize": "0.82rem"}),
            ])
            for label, val, color in rows
        ])
    ], style={"width": "100%"})



# PAGE 2 — REVENUE INTELLIGENCE


def page_revenue():
    data = api_get("/insights/revenue") or []
    if not data:
        return html.P("Could not load revenue data.", style={"color": COLORS["danger"]})
    df = pd.DataFrame(data)

    # MoM growth
    df["growth"] = df["revenue"].pct_change() * 100

    # Peak month
    peak_idx = df["revenue"].idxmax()

    # Chart 1: Revenue bars
    fig_rev = go.Figure()
    colors_bar = [COLORS["warning"] if i == peak_idx else COLORS["primary"]
                  for i in range(len(df))]
    fig_rev.add_trace(go.Bar(
        x=df["month_label"], y=df["revenue"],
        marker_color=colors_bar,
        name="Gross Revenue",
        hovertemplate="<b>%{x}</b><br>Revenue: R$%{y:,.0f}<extra></extra>",
    ))
    fig_rev.update_layout(
        **CHART_LAYOUT,
        title="Monthly Gross Revenue (BRL) — Delivered Orders",
        yaxis_tickformat="R$,.0f",
        showlegend=False,
        annotations=[dict(
            x=df.loc[peak_idx, "month_label"],
            y=df.loc[peak_idx, "revenue"],
            text="⭐ Peak",
            showarrow=True, arrowhead=2,
            arrowcolor=COLORS["warning"],
            font=dict(color=COLORS["warning"], size=12),
        )],
    )

    # Chart 2: MoM growth
    fig_growth = go.Figure()
    growth_colors = [COLORS["success"] if g >= 0 else COLORS["danger"]
                     for g in df["growth"].fillna(0)]
    fig_growth.add_trace(go.Bar(
        x=df["month_label"], y=df["growth"],
        marker_color=growth_colors,
        name="MoM Growth",
        hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}%<extra></extra>",
    ))
    fig_growth.update_layout(
        **CHART_LAYOUT,
        title="Month-over-Month Revenue Growth (%)",
        yaxis_ticksuffix="%",
        showlegend=False,
    )
    fig_growth.add_hline(y=0, line_color=COLORS["muted"], line_dash="dot")

    # Chart 3: Orders vs Customers
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df["month_label"], y=df["orders"],
        mode="lines+markers", name="Orders",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=5),
    ))
    fig_vol.add_trace(go.Scatter(
        x=df["month_label"], y=df["customers"],
        mode="lines+markers", name="Unique Customers",
        line=dict(color=COLORS["success"], width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig_vol.update_layout(
        **CHART_LAYOUT,
        title="Orders vs Unique Customers per Month",
        yaxis_tickformat=",",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )

    # Chart 4: AOV + Rating dual axis
    fig_aov = go.Figure()
    fig_aov.add_trace(go.Bar(
        x=df["month_label"], y=df["aov"],
        name="Avg Order Value (BRL)",
        marker_color=COLORS["orange"],
        opacity=0.75,
        yaxis="y",
    ))
    fig_aov.add_trace(go.Scatter(
        x=df["month_label"], y=df["avg_rating"],
        name="Avg Rating",
        line=dict(color=COLORS["purple"], width=2.5),
        marker=dict(size=6, symbol="diamond"),
        yaxis="y2",
        mode="lines+markers",
    ))
    fig_aov.update_layout(
        **CHART_LAYOUT,
        title="Avg Order Value vs Customer Rating",
        yaxis=dict(title="AOV (BRL)", gridcolor=COLORS["border"],
                   linecolor=COLORS["border"]),
        yaxis2=dict(title="Avg Rating", overlaying="y", side="right",
                    range=[3.5, 5.0], gridcolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )

    return html.Div([
        section_header("Revenue Intelligence",
                       "Monthly revenue trends, growth rates, and order volume analysis"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_rev,    config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_growth, config={"displayModeBar": False}), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_vol,    config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_aov,    config={"displayModeBar": False}), md=6),
        ]),
    ])



# PAGE 3 — CUSTOMER SEGMENTS


def page_segments():
    data = api_get("/insights/segments") or []
    if not data:
        return html.P("Could not load segment data.", style={"color": COLORS["danger"]})
    df = pd.DataFrame(data)

    seg_colors = {
        "Champions":       COLORS["success"],
        "Loyal Customers": COLORS["primary"],
        "Promising":       "#39d353",
        "Needs Attention": COLORS["warning"],
        "At Risk":         COLORS["orange"],
        "Lost":            COLORS["danger"],
    }
    colors = [seg_colors.get(s, COLORS["primary"]) for s in df["segment"]]

    # Donut
    fig_donut = go.Figure(go.Pie(
        labels=df["segment"],
        values=df["customer_count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig_donut.update_layout(
        **CHART_LAYOUT,
        title="Customer Segments — RFM Analysis",
        annotations=[dict(
            text=f"<b>{df['customer_count'].sum():,}</b><br><span style='font-size:10px'>Customers</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text"]),
        )],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Horizontal bar — segment count
    fig_bar = go.Figure(go.Bar(
        y=df["segment"],
        x=df["customer_count"],
        orientation="h",
        marker_color=colors,
        text=df["pct"].apply(lambda p: f"{p}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
    ))
    fig_bar.update_layout(
        **CHART_LAYOUT,
        title="Customer Count by Segment",
        xaxis_tickformat=",",
        margin=dict(l=120, r=60, t=40, b=40),
    )

    # Action table
    actions = {
        "Champions":       "🎁 Reward with referral programme",
        "Loyal Customers": "⬆️  Upsell premium products",
        "Promising":       "🎯 First-time offer to convert",
        "Needs Attention": "📧 Nurture with personalised content",
        "At Risk":         "🚨 Win-back campaign immediately",
        "Lost":            "📋 Exit survey — learn why they left",
    }
    table_rows = []
    for _, row in df.iterrows():
        table_rows.append(html.Tr([
            html.Td(row["segment"],
                    style={"color": seg_colors.get(row["segment"], COLORS["text"]),
                           "fontWeight": "600", "padding": "8px 12px"}),
            html.Td(f"{int(row['customer_count']):,}",
                    style={"color": COLORS["text"], "padding": "8px 12px"}),
            html.Td(f"{float(row['pct']):.1f}%",
                    style={"color": COLORS["muted"], "padding": "8px 12px"}),
            html.Td(actions.get(row["segment"], "—"),
                    style={"color": COLORS["muted"], "fontSize": "0.82rem",
                           "padding": "8px 12px"}),
        ]))

    action_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Segment",  style={"padding": "8px 12px", "color": COLORS["muted"],
                                       "fontWeight": "600", "fontSize": "0.78rem"}),
            html.Th("Count",    style={"padding": "8px 12px", "color": COLORS["muted"],
                                       "fontWeight": "600", "fontSize": "0.78rem"}),
            html.Th("%",        style={"padding": "8px 12px", "color": COLORS["muted"],
                                       "fontWeight": "600", "fontSize": "0.78rem"}),
            html.Th("Recommended Action", style={"padding": "8px 12px",
                                                  "color": COLORS["muted"],
                                                  "fontWeight": "600",
                                                  "fontSize": "0.78rem"}),
        ])),
        html.Tbody(table_rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    return html.Div([
        section_header("Customer Segments",
                       "RFM segmentation — Recency, Frequency, Monetary value analysis"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_donut, config={"displayModeBar": False}), md=5),
            dbc.Col(dcc.Graph(figure=fig_bar,   config={"displayModeBar": False}), md=7),
        ], className="mb-4"),
        dbc.Card(dbc.CardBody([
            html.H6("Segment Actions", style={"color": COLORS["primary"],
                                               "fontWeight": "700", "marginBottom": "12px"}),
            action_table,
        ]), style={"backgroundColor": COLORS["card"],
                   "border": f"1px solid {COLORS['border']}",
                   "borderRadius": "8px"}),
    ])



# PAGE 4 — DELIVERY IMPACT


def page_delivery():
    data = api_get("/insights/delivery") or []
    cat_data = api_get("/insights/top-categories") or []

    if not data:
        return html.P("Could not load delivery data.", style={"color": COLORS["danger"]})

    df_del = pd.DataFrame(data)
    df_cat = pd.DataFrame(cat_data) if cat_data else pd.DataFrame()

    bucket_order = ["On Time", "Late: 0-1d", "Late: 1-3d", "Late: 3-7d", "Late: 7d+"]
    df_del = df_del[df_del["delivery_bucket"].isin(bucket_order)]
    df_del["delivery_bucket"] = pd.Categorical(df_del["delivery_bucket"],
                                                categories=bucket_order, ordered=True)
    df_del = df_del.sort_values("delivery_bucket")

    # Chart 1: Avg rating by delivery bucket
    bar_colors = [COLORS["success"] if b == "On Time" else COLORS["danger"]
                  for b in df_del["delivery_bucket"]]
    fig_rating = go.Figure(go.Bar(
        x=df_del["delivery_bucket"],
        y=df_del["avg_rating"],
        marker_color=bar_colors,
        text=df_del["avg_rating"].apply(lambda v: f"⭐ {v}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Rating: %{y:.2f}<extra></extra>",
    ))
    fig_rating.update_layout(
        **CHART_LAYOUT,
        title="Average Review Rating by Delivery Status",
        yaxis=dict(range=[1, 5.5], title="Avg Rating",
                   gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        showlegend=False,
        annotations=[dict(
            x="Late: 0-1d",
            y=df_del[df_del["delivery_bucket"] == "Late: 0-1d"]["avg_rating"].values[0]
            if "Late: 0-1d" in df_del["delivery_bucket"].values else 2.5,
            text="⚡ Even 1 day late<br>tanks satisfaction",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS["warning"],
            font=dict(color=COLORS["warning"], size=11),
            ax=-60, ay=-50,
        )],
    )

    # Chart 2: Order volume by bucket
    fig_vol = go.Figure(go.Bar(
        x=df_del["delivery_bucket"],
        y=df_del["order_count"],
        marker_color=bar_colors,
        text=df_del["pct_of_orders"].apply(lambda p: f"{p}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Orders: %{value:,}<extra></extra>",
    ))
    fig_vol.update_layout(
        **CHART_LAYOUT,
        title="Order Volume by Delivery Status",
        yaxis=dict(title="Order Count", tickformat=",",
                   gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        showlegend=False,
    )

    # Chart 3: Top categories
    fig_cat = go.Figure()
    if not df_cat.empty:
        df_cat = df_cat.head(12).sort_values("revenue")
        cat_colors = [
            COLORS["warning"] if r == df_cat["revenue"].max()
            else COLORS["primary"]
            for r in df_cat["revenue"]
        ]
        fig_cat.add_trace(go.Bar(
            y=df_cat["category"],
            x=df_cat["revenue"],
            orientation="h",
            marker_color=cat_colors,
            hovertemplate="<b>%{y}</b><br>Revenue: R$%{x:,.0f}<extra></extra>",
        ))
    fig_cat.update_layout(
        **CHART_LAYOUT,
        title="Top 12 Categories by Revenue",
        xaxis=dict(title="Revenue (BRL)", tickformat="R$,.0f",
                   gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        margin=dict(l=160, r=20, t=40, b=40),
        showlegend=False,
    )

    # Key insight card
    on_time_rating = df_del[df_del["delivery_bucket"] == "On Time"]["avg_rating"].values
    late_7_rating  = df_del[df_del["delivery_bucket"] == "Late: 7d+"]["avg_rating"].values
    drop = round(float(on_time_rating[0]) - float(late_7_rating[0]), 2) if (
        len(on_time_rating) and len(late_7_rating)
    ) else 0

    insight_card = dbc.Card(dbc.CardBody([
        html.H6("💡 Key Business Insight",
                style={"color": COLORS["warning"], "fontWeight": "700"}),
        html.P([
            f"Late delivery 7+ days causes avg rating to drop ",
            html.Strong(f"{drop} points", style={"color": COLORS["danger"]}),
            " vs on-time delivery. ",
            html.Br(),
            "Business action: Tighten SLA for heavy/bulky categories — "
            "this single change reduces churn risk without discounting.",
        ], style={"color": COLORS["text"], "fontSize": "0.88rem", "marginBottom": "0"}),
    ]), style={"backgroundColor": COLORS["card"],
               "border": f"1px solid {COLORS['warning']}",
               "borderRadius": "8px"})

    return html.Div([
        section_header("Delivery Impact",
                       "How delivery performance drives customer satisfaction and repeat purchases"),
        dbc.Row([insight_card], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_rating, config={"displayModeBar": False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_vol,    config={"displayModeBar": False}), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cat, config={"displayModeBar": False}), md=12),
        ]),
    ])



# PAGE 5 — CHURN PREDICTOR (Interactive)


def page_predict():
    input_style = {
        "backgroundColor": COLORS["bg"],
        "border":          f"1px solid {COLORS['border']}",
        "color":           COLORS["text"],
        "borderRadius":    "6px",
        "padding":         "8px 12px",
        "width":           "100%",
    }
    label_style = {
        "color":      COLORS["muted"],
        "fontSize":   "0.78rem",
        "fontWeight": "600",
        "marginBottom": "4px",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
    }

    def make_input(id_, placeholder="", type_="number", value=None, step=None):
        props = dict(
            id=id_, type=type_, placeholder=placeholder,
            value=value, style=input_style,
            debounce=True,
        )
        if step:
            props["step"] = step
        return dcc.Input(**props)

    def make_select(id_, options, value):
        return dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in options],
            value=value,
            style={
                "backgroundColor": COLORS["bg"],
                "color":           COLORS["bg"],
                "border":          f"1px solid {COLORS['border']}",
                "borderRadius":    "6px",
            },
            className="dark-dropdown",
        )

    form = dbc.Card(dbc.CardBody([
        html.H6("First Order Details", style={"color": COLORS["primary"],
                                               "fontWeight": "700",
                                               "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col([html.P("Order Value (BRL)", style=label_style),
                     make_input("inp-order-value", "e.g. 189.90", value=189.90, step=0.01)], md=3),
            dbc.Col([html.P("Freight Value (BRL)", style=label_style),
                     make_input("inp-freight", "e.g. 18.50", value=18.50, step=0.01)], md=3),
            dbc.Col([html.P("Item Count", style=label_style),
                     make_input("inp-items", "e.g. 2", value=2)], md=3),
            dbc.Col([html.P("Avg Item Price (BRL)", style=label_style),
                     make_input("inp-avg-price", "e.g. 94.95", value=94.95, step=0.01)], md=3),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([html.P("Actual Delivery Days", style=label_style),
                     make_input("inp-actual-days", "e.g. 8", value=8, step=0.5)], md=3),
            dbc.Col([html.P("Promised Delivery Days", style=label_style),
                     make_input("inp-promised-days", "e.g. 12", value=12, step=0.5)], md=3),
            dbc.Col([html.P("Delay Hours (0 if on time)", style=label_style),
                     make_input("inp-delay", "e.g. 0", value=0, step=1)], md=3),
            dbc.Col([html.P("Delivered On Time?", style=label_style),
                     make_select("inp-on-time", ["1 — Yes", "0 — No"], "1 — Yes")], md=3),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([html.P("Review Score (1–5)", style=label_style),
                     make_input("inp-review", "e.g. 5", value=5, step=1)], md=3),
            dbc.Col([html.P("Payment Type", style=label_style),
                     make_select("inp-payment",
                                 ["credit_card", "boleto", "voucher", "debit_card"],
                                 "credit_card")], md=3),
            dbc.Col([html.P("Installments", style=label_style),
                     make_input("inp-installments", "e.g. 3", value=3)], md=3),
            dbc.Col([html.P("Product Category", style=label_style),
                     make_select("inp-category", [
                         "health_beauty", "bed_bath_table", "sports_leisure",
                         "furniture_decor", "computers_accessories", "housewares",
                         "watches_gifts", "telephony", "auto", "toys",
                         "cool_stuff", "garden_tools", "baby", "electronics",
                         "fashion_bags_accessories", "stationery", "other",
                     ], "health_beauty")], md=3),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([html.P("Product Weight (g)", style=label_style),
                     make_input("inp-weight", "e.g. 400", value=400)], md=3),
            dbc.Col([html.P("Purchase Month (1–12)", style=label_style),
                     make_input("inp-month", "e.g. 11", value=11)], md=3),
            dbc.Col([html.P("Purchase Day of Week (0=Mon)", style=label_style),
                     make_input("inp-dow", "e.g. 1", value=1)], md=3),
            dbc.Col([html.P("Customer State", style=label_style),
                     make_select("inp-state", [
                         "SP", "RJ", "MG", "RS", "PR", "SC",
                         "BA", "GO", "ES", "PE", "other",
                     ], "SP")], md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dbc.Button("🔮 Predict Churn Risk", id="btn-predict",
                               color="primary", size="lg",
                               style={"width": "100%", "fontWeight": "700"})),
        ]),
    ]), style={"backgroundColor": COLORS["card"],
               "border":         f"1px solid {COLORS['border']}",
               "borderRadius":   "8px"})

    result_panel = html.Div(id="predict-result", style={"marginTop": "24px"})

    return html.Div([
        section_header("Churn Predictor",
                       "Enter a customer's first-order details to predict if they'll return"),
        form,
        result_panel,
    ])



# CALLBACKS


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname in (None, "/", ""):
        return page_summary()
    elif pathname == "/revenue":
        return page_revenue()
    elif pathname == "/segments":
        return page_segments()
    elif pathname == "/delivery":
        return page_delivery()
    elif pathname == "/predict":
        return page_predict()
    return html.P("Page not found.", style={"color": COLORS["danger"]})


@app.callback(
    Output("predict-result", "children"),
    Input("btn-predict", "n_clicks"),
    [
        State("inp-order-value",    "value"),
        State("inp-freight",        "value"),
        State("inp-items",          "value"),
        State("inp-avg-price",      "value"),
        State("inp-actual-days",    "value"),
        State("inp-promised-days",  "value"),
        State("inp-delay",          "value"),
        State("inp-on-time",        "value"),
        State("inp-review",         "value"),
        State("inp-payment",        "value"),
        State("inp-installments",   "value"),
        State("inp-category",       "value"),
        State("inp-weight",         "value"),
        State("inp-month",          "value"),
        State("inp-dow",            "value"),
        State("inp-state",          "value"),
    ],
    prevent_initial_call=True,
)
def run_prediction(n_clicks, order_val, freight, items, avg_price,
                   actual_days, promised_days, delay, on_time,
                   review, payment, installments, category,
                   weight, month, dow, state):
    if not n_clicks:
        return html.Div()

    # Parse on_time select value ("1 — Yes" → 1)
    on_time_int = 1 if str(on_time).startswith("1") else 0

    # Build derived features
    freight_pct = round((float(freight or 0) / max(float(order_val or 1), 0.01)) * 100, 2)
    speed_ratio = round(float(actual_days or 1) / max(float(promised_days or 1), 0.01), 4)
    positive_review = 1 if int(review or 3) >= 4 else 0
    bad_review      = 1 if int(review or 3) <= 2 else 0

    payload = {
        "order_value":            float(order_val or 0),
        "freight_value":          float(freight or 0),
        "freight_pct":            freight_pct,
        "item_count":             int(items or 1),
        "avg_item_price":         float(avg_price or 0),
        "actual_delivery_days":   float(actual_days or 0),
        "promised_delivery_days": float(promised_days or 1),
        "delay_hours":            float(delay or 0),
        "delivered_on_time":      on_time_int,
        "delivery_speed_ratio":   speed_ratio,
        "review_score":           float(review or 3),
        "positive_review":        positive_review,
        "bad_review":             bad_review,
        "review_response_days":   0.0,
        "payment_type":           payment or "credit_card",
        "installments":           int(installments or 1),
        "payment_methods_used":   1,
        "category_grouped":       category or "other",
        "product_weight_g":       float(weight or 0),
        "product_photos_qty":     0,
        "purchase_month":         int(month or 1),
        "purchase_dow":           int(dow or 0),
        "purchase_hour":          12,
        "state_grouped":          state or "other",
    }

    try:
        resp = requests.post(f"{API_BASE}/predict/churn", json=payload, timeout=8)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return dbc.Alert(f"Prediction failed: {e}", color="danger")

    prob   = result["churn_probability"]
    label  = result["risk_label"]
    score  = result["risk_score"]
    rec    = result["recommendation"]
    factors= result["top_risk_factors"]

    # Gauge chart
    label_color = {
        "HIGH":   COLORS["danger"],
        "MEDIUM": COLORS["orange"],
        "LOW":    COLORS["warning"],
        "SAFE":   COLORS["success"],
    }.get(label, COLORS["primary"])

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50, "valueformat": ".0f"},
        number={"suffix": "%", "font": {"color": label_color, "size": 36}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": COLORS["text"]},
            "bar":   {"color": label_color, "thickness": 0.25},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["border"],
            "steps": [
                {"range": [0, 25],  "color": "rgba(63,185,80,0.15)"},
                {"range": [25, 50], "color": "rgba(210,153,34,0.15)"},
                {"range": [50, 75], "color": "rgba(255,166,87,0.15)"},
                {"range": [75, 100],"color": "rgba(248,81,73,0.15)"},
            ],
            "threshold": {
                "line": {"color": COLORS["warning"], "width": 3},
                "thickness": 0.75,
                "value": 36,
            },
        },
        title={"text": "Churn Risk Score", "font": {"color": COLORS["text"]}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        height=260,
        margin=dict(l=30, r=30, t=30, b=10),
    )

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            dcc.Graph(figure=fig_gauge, config={"displayModeBar": False}),
        ]), style={"backgroundColor": COLORS["card"],
                   "border": f"2px solid {label_color}",
                   "borderRadius": "8px"}), md=4),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(label, style={"color": label_color, "fontWeight": "700",
                                   "fontSize": "2rem", "marginBottom": "4px"}),
            html.P(f"Churn probability: {prob:.1%}",
                   style={"color": COLORS["text"], "fontSize": "1rem"}),
            html.Hr(style={"borderColor": COLORS["border"]}),
            html.H6("Recommendation", style={"color": COLORS["muted"],
                                              "textTransform": "uppercase",
                                              "fontSize": "0.75rem",
                                              "letterSpacing": "1px"}),
            html.P(rec, style={"color": COLORS["text"], "fontSize": "0.95rem",
                                "marginBottom": "16px"}),
            html.H6("Risk Factors Detected",
                    style={"color": COLORS["muted"],
                           "textTransform": "uppercase",
                           "fontSize": "0.75rem",
                           "letterSpacing": "1px"}),
            html.Ul([html.Li(f, style={"color": COLORS["text"], "marginBottom": "4px"})
                     for f in factors],
                    style={"paddingLeft": "18px", "fontSize": "0.88rem"}),
        ]), style={"backgroundColor": COLORS["card"],
                   "border": f"1px solid {COLORS['border']}",
                   "borderRadius": "8px",
                   "height": "100%"}), md=8),
    ], className="g-3")



# MAIN

if __name__ == "__main__":
    app.run(debug=True, port=8050)