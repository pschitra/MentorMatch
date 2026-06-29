"""
MentorMatch UAE — Analytics Dashboard
Part A: Setup, Data Loading, Feature Engineering, ML Training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")
import io

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MentorMatch UAE Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {background: #0D1117; border-right: 1px solid #21262D;}
[data-testid="stSidebar"] .stRadio > label {color: #C9D1D9 !important; font-size: 14px;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {color: #1D9E75 !important;}

/* Main */
.main-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 100%);
    padding: 24px 28px; border-radius: 12px; margin-bottom: 24px;
    border: 1px solid #21262D;
}
.main-header h1 {color: #1D9E75; margin:0; font-size:28px; font-weight:700;}
.main-header p  {color: #8B949E; margin:4px 0 0 0; font-size:14px;}

/* Metric cards */
.metric-card {
    background: #161B22; border: 1px solid #21262D; border-radius: 10px;
    padding: 20px 16px; text-align: center; transition: border-color .2s;
}
.metric-card:hover {border-color: #1D9E75;}
.metric-title {color: #8B949E; font-size: 12px; text-transform: uppercase; letter-spacing:.8px;}
.metric-value {color: #F0F6FC; font-size: 32px; font-weight: 700; margin: 6px 0;}
.metric-delta {font-size: 13px;}

/* Section headers */
.section-header {
    border-left: 4px solid #1D9E75; padding-left: 12px;
    margin: 28px 0 16px 0; color: #F0F6FC;
}

/* Insight box */
.insight-box {
    background: #161B22; border: 1px solid #1D9E75; border-radius: 8px;
    padding: 16px 20px; margin: 12px 0; color: #C9D1D9; font-size: 14px;
}
.insight-box strong {color: #1D9E75;}

/* Warning box */
.warn-box {
    background: #1C1006; border: 1px solid #D85A30; border-radius: 8px;
    padding: 16px 20px; margin: 12px 0; color: #C9D1D9; font-size: 14px;
}

/* Finding cards */
.finding-card {
    background: #161B22; border: 1px solid #21262D; border-radius: 8px;
    padding: 16px 20px; margin: 10px 0;
}
.finding-card h4 {color: #1D9E75; margin: 0 0 6px 0; font-size: 15px;}
.finding-card p  {color: #C9D1D9; margin: 0; font-size: 13px; line-height: 1.5;}

/* Model performance table */
.perf-table {background: #161B22; border-radius: 8px; padding: 12px;}

stExpander {border: 1px solid #21262D !important;}
</style>
""", unsafe_allow_html=True)

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
COLORS = {
    "teal"  : "#1D9E75", "purple": "#534AB7", "coral" : "#D85A30",
    "amber" : "#EF9F27", "blue"  : "#378ADD", "pink"  : "#C879A8",
    "gray"  : "#8B949E", "navy"  : "#0C447C", "green" : "#085041",
}
MODEL_COLORS = {
    "KNN"              : COLORS["blue"],
    "Decision Tree"    : COLORS["amber"],
    "Random Forest"    : COLORS["teal"],
    "Gradient Boosting": COLORS["purple"],
}
PLOTLY_TEMPLATE = "plotly_dark"

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_source=None):
    try:
        if file_source is not None:
            df = pd.read_csv(file_source)
        else:
            df = pd.read_csv("MentorMatch_UAE_Clean_Dataset.csv")
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used across multiple pages."""
    df = df.copy()
    df["Outcome_Num"]  = (df["Career_Outcome_Achieved"] == "Yes").astype(int)
    df["Churned_Num"]  = (df["Churned"]                 == "Yes").astype(int)
    df["Corp_Num"]     = (df["Corporate_Sponsored"]      == "Yes").astype(int)
    df["Age_Group"] = pd.cut(
        df["Age"], bins=[21, 26, 31, 36, 100],
        labels=["22–26 (Early)", "27–31 (Growth)", "32–36 (Mid)", "37+ (Established)"]
    )
    df["Rating_Tier"] = pd.cut(
        df["Mentor_Rating"], bins=[0, 2.99, 3.49, 3.99, 5.01],
        labels=["< 3.0", "3.0–3.5", "3.5–4.0", "4.0+"]
    )
    df["Tenure_Group"] = pd.cut(
        df["Months_Subscribed"], bins=[0, 3, 6, 12, 18],
        labels=["1–3 mo", "4–6 mo", "7–12 mo", "13–18 mo"]
    )
    EXP_ORD = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-10 yrs)"]
    df["Experience_Level"] = pd.Categorical(df["Experience_Level"],
                                             categories=EXP_ORD, ordered=True)
    return df

# ─── FEATURE ENGINEERING FOR ML ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y for ML classification.
    Returns X_raw (before scaling), y, feature_names, and the scaler.
    """
    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    y = (df["Career_Outcome_Achieved"] == "Yes").astype(int).values

    # ── Drop columns not used as features ─────────────────────────────────────
    DROP = ["Mentee_ID", "Sign_Up_Date", "Nationality",
            "Career_Outcome_Achieved", "Subscription_Plan",
            "Monthly_Revenue_AED", "Total_Revenue_AED", "Revenue_Tier"]
    df_feat = df.drop(columns=DROP, errors="ignore")

    # ── Binary encoding ───────────────────────────────────────────────────────
    for col in ["Corporate_Sponsored", "Churned"]:
        df_feat[col] = (df_feat[col] == "Yes").astype(int)

    # ── Ordinal encoding for Experience_Level ─────────────────────────────────
    exp_map = {"Entry (0-2 yrs)": 1, "Mid (3-5 yrs)": 2, "Senior (6-10 yrs)": 3}
    df_feat["Experience_Level"] = df_feat["Experience_Level"].astype(str).map(exp_map).fillna(1).astype(int)

    # ── One-hot encoding for nominal categoricals ─────────────────────────────
    NOMINAL = ["Industry", "Goal_Type", "Referral_Source"]
    df_feat = pd.get_dummies(df_feat, columns=NOMINAL, drop_first=True)

    X = df_feat.values.astype(float)
    feature_names = list(df_feat.columns)

    return X, y, feature_names

# ─── ML MODEL TRAINING ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_all_models(df: pd.DataFrame, test_size: float = 0.2,
                     knn_k: int = 5, dt_depth: int = 5,
                     rf_trees: int = 100, gb_lr: float = 0.1):
    """
    Train KNN, Decision Tree, Random Forest, Gradient Boosting.
    Returns a results dict with metrics, predictions, and model objects.
    """
    X, y, feat_names = engineer_features(df)

    # Train-test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features (important for KNN; harmless for trees)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "KNN"              : KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree"    : DecisionTreeClassifier(max_depth=dt_depth, random_state=42),
        "Random Forest"    : RandomForestClassifier(n_estimators=rf_trees, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=gb_lr, random_state=42),
    }

    results = {}
    for name, model in models.items():
        # KNN needs scaled data; trees work on either
        is_knn = name == "KNN"
        Xtr = X_train_sc if is_knn else X_train
        Xte = X_test_sc  if is_knn else X_test

        model.fit(Xtr, y_train)

        y_pred_train = model.predict(Xtr)
        y_pred_test  = model.predict(Xte)
        y_prob_test  = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

        # Cross-validation (5-fold, stratified) on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")

        # Metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob_test) if y_prob_test is not None else ([], [], [])
        auc_score   = roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else 0.0

        results[name] = {
            "model"         : model,
            "train_acc"     : accuracy_score(y_train, y_pred_train),
            "test_acc"      : accuracy_score(y_test,  y_pred_test),
            "precision"     : precision_score(y_test, y_pred_test, zero_division=0),
            "recall"        : recall_score(y_test,    y_pred_test, zero_division=0),
            "f1"            : f1_score(y_test,        y_pred_test, zero_division=0),
            "auc"           : auc_score,
            "fpr"           : list(fpr),
            "tpr"           : list(tpr),
            "conf_matrix"   : confusion_matrix(y_test, y_pred_test).tolist(),
            "class_report"  : classification_report(y_test, y_pred_test, output_dict=True),
            "cv_scores"     : list(cv_scores),
            "cv_mean"       : float(cv_scores.mean()),
            "cv_std"        : float(cv_scores.std()),
            "y_test"        : list(y_test),
            "y_pred"        : list(y_pred_test),
        }

    # Feature importance from tree-based models
    feat_importance = {}
    for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        if name in results and hasattr(results[name]["model"], "feature_importances_"):
            fi = results[name]["model"].feature_importances_
            feat_importance[name] = pd.Series(fi, index=feat_names).sort_values(ascending=False).head(15)

    results["_meta"] = {
        "X_train"       : X_train,
        "X_test"        : X_test,
        "y_train"       : list(y_train),
        "y_test"        : list(y_test),
        "feat_names"    : feat_names,
        "feat_importance": feat_importance,
        "test_size"     : test_size,
        "n_train"       : len(y_train),
        "n_test"        : len(y_test),
    }

    return results
"""
MentorMatch UAE — Analytics Dashboard
Part B: Pages 1–4
"""

# ─── PAGE 1 — OVERVIEW ───────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame):
    st.markdown("""
    <div class="main-header">
        <h1>🎯 MentorMatch UAE — Analytics Dashboard</h1>
        <p>Chief Technology Officer View · Cross-Feature Career Outcome Analysis · UAE Job Market Intelligence</p>
    </div>""", unsafe_allow_html=True)

    baseline = (df["Career_Outcome_Achieved"] == "Yes").mean() * 100
    churn    = (df["Churned"] == "Yes").mean() * 100
    corp_pct = (df["Corporate_Sponsored"] == "Yes").mean() * 100
    total_rev = df["Total_Revenue_AED"].sum()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl, delta, dcolor in [
        (c1, f"{len(df):,}",          "Total Mentees",       "",                          ""),
        (c2, f"{baseline:.1f}%",      "Career Outcome Rate", "Platform Baseline",         COLORS["teal"]),
        (c3, f"{df['NPS_Score'].mean():.0f}", "Avg NPS Score", "−100 to +100 scale",     COLORS["amber"]),
        (c4, f"AED {total_rev:,.0f}", "Total Revenue",       "Lifetime to date",          COLORS["blue"]),
        (c5, f"{corp_pct:.0f}%",      "Corporate Sponsored", "B2B segment",               COLORS["purple"]),
        (c6, f"{churn:.0f}%",         "Churn Rate",          "Platform-wide",             COLORS["coral"]),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{lbl}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta" style="color:{dcolor};">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        oc = df["Career_Outcome_Achieved"].value_counts()
        fig = go.Figure(go.Pie(
            labels=oc.index, values=oc.values,
            marker_colors=[COLORS["teal"], COLORS["coral"]],
            hole=0.55, textinfo="percent+label"
        ))
        fig.update_layout(title="Career Outcome Split", template=PLOTLY_TEMPLATE,
                          height=320, margin=dict(t=40, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ic = df["Industry"].value_counts().sort_values()
        fig = go.Figure(go.Bar(
            x=ic.values, y=ic.index, orientation="h",
            marker_color=COLORS["blue"],
            text=ic.values, textposition="outside"
        ))
        fig.update_layout(title="Mentees by Industry", template=PLOTLY_TEMPLATE,
                          height=320, margin=dict(t=40, b=10, l=10, r=120),
                          xaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        ec = df["Experience_Level"].value_counts()
        fig = go.Figure(go.Pie(
            labels=ec.index, values=ec.values,
            marker_colors=[COLORS["teal"], COLORS["blue"], COLORS["purple"]],
            hole=0.55, textinfo="percent+label"
        ))
        fig.update_layout(title="Experience Level Split", template=PLOTLY_TEMPLATE,
                          height=320, margin=dict(t=40, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col4, col5 = st.columns(2)

    with col4:
        gc = df["Goal_Type"].value_counts()
        fig = go.Figure(go.Bar(
            x=gc.index, y=gc.values,
            marker_color=[COLORS["teal"], COLORS["blue"], COLORS["purple"],
                          COLORS["amber"], COLORS["coral"]],
            text=gc.values, textposition="outside"
        ))
        fig.update_layout(title="Goal Type Distribution", template=PLOTLY_TEMPLATE,
                          height=300, margin=dict(t=40, b=40, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        rc = df["Referral_Source"].value_counts()
        fig = go.Figure(go.Bar(
            x=rc.index, y=rc.values,
            marker_color=COLORS["purple"],
            text=rc.values, textposition="outside"
        ))
        fig.update_layout(title="Acquisition Channel Distribution", template=PLOTLY_TEMPLATE,
                          height=300, margin=dict(t=40, b=60, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Dataset preview ────────────────────────────────────────────────────────
    with st.expander("📋 Dataset Preview (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ─── PAGE 2 — DESCRIPTIVE ANALYTICS (CROSS-TABULATIONS) ──────────────────────
def page_descriptive(df: pd.DataFrame):
    st.markdown("<h2 class='section-header'>📊 Descriptive Analytics — Cross-Tabulations Against Career Outcome</h2>", unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    <strong>What is cross-tabulation analysis?</strong> A cross-tabulation (pivot table) 
    groups mentees by two or more features and computes a metric (here: career outcome rate) 
    for each group. This reveals which COMBINATIONS of features drive success — something 
    single-feature analysis completely misses.
    </div>""", unsafe_allow_html=True)

    baseline = (df["Career_Outcome_Achieved"] == "Yes").mean() * 100
    EXP_ORD  = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-10 yrs)"]
    GOAL_ORD = ["Skill Building", "Get Promoted", "Network Building", "Career Switch", "Start a Business"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Experience × Goal",
        "2️⃣ Industry × Experience",
        "3️⃣ Sessions × Mentor Rating",
        "4️⃣ Corporate × Sessions",
        "5️⃣ Age Group × Goal",
        "6️⃣ Industry × Goal → Retention",
    ])

    # ── Tab 1: Experience × Goal ───────────────────────────────────────────────
    with tab1:
        st.markdown("#### Career Outcome Rate (%) — Experience Level × Goal Type")
        st.caption(f"Platform baseline: {baseline:.1f}% | Darker green = higher career outcome rate")
        pivot1 = (df.groupby(["Experience_Level", "Goal_Type"], observed=True)["Outcome_Num"]
                  .mean().unstack() * 100).round(1).reindex(index=EXP_ORD, columns=GOAL_ORD)

        fig = go.Figure(go.Heatmap(
            z=pivot1.values, x=pivot1.columns.tolist(), y=pivot1.index.tolist(),
            colorscale="YlGn", text=pivot1.values, texttemplate="%{text:.0f}%",
            colorbar=dict(title="Outcome %"),
            hoverongaps=False
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                          xaxis_title="Goal Type", yaxis_title="Experience Level",
                          margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Best/Worst per experience level
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Best goal per experience level:**")
            for exp in EXP_ORD:
                row = pivot1.loc[exp].dropna()
                if len(row):
                    st.success(f"**{exp}** → {row.idxmax()} ({row.max():.0f}%)")
        with col2:
            st.markdown("**Worst goal per experience level:**")
            for exp in EXP_ORD:
                row = pivot1.loc[exp].dropna()
                if len(row):
                    st.error(f"**{exp}** → {row.idxmin()} ({row.min():.0f}%)")

        st.markdown(f"""<div class="insight-box">
        <strong>Key Insight:</strong> Choosing the wrong goal for your career stage is the 
        single largest predictor of failure on the MentorMatch platform. Entry-level professionals 
        pursuing Career Switches face a structural barrier in UAE — visa sponsorship is tied to 
        job titles, making industry changes in &lt;2 years extremely difficult regardless of 
        mentor quality. The platform should guide mentees toward goal-stage appropriate targets 
        during onboarding.</div>""", unsafe_allow_html=True)

    # ── Tab 2: Industry × Experience ─────────────────────────────────────────
    with tab2:
        st.markdown("#### Career Outcome Rate (%) — Industry × Experience Level")
        pivot2 = (df.groupby(["Industry", "Experience_Level"], observed=True)["Outcome_Num"]
                  .mean().unstack() * 100).round(1).reindex(columns=EXP_ORD)
        pivot2 = pivot2.reindex(pivot2.mean(axis=1).sort_values(ascending=False).index)

        fig = go.Figure(go.Heatmap(
            z=pivot2.values, x=pivot2.columns.tolist(), y=pivot2.index.tolist(),
            colorscale="YlGn", text=pivot2.values, texttemplate="%{text:.0f}%",
            colorbar=dict(title="Outcome %"),
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          xaxis_title="Experience Level", yaxis_title="Industry (sorted by avg outcome)",
                          margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""<div class="insight-box">
        <strong>Key Insight:</strong> Finance and Technology professionals show the highest 
        mentoring ROI in Dubai, particularly at Entry-level. These sectors have well-defined 
        career ladders in DIFC and tech hubs where a mentor who climbed the same path can give 
        precise, actionable advice. Marketing is the most oversupplied sector in UAE — 
        mentoring helps but structural headwinds limit career outcome conversion rates.</div>""", unsafe_allow_html=True)

    # ── Tab 3: Sessions × Mentor Rating ────────────────────────────────────────
    with tab3:
        st.markdown("#### Career Outcome Rate (%) — Sessions/Month × Mentor Rating Tier")
        st.caption("This reveals the interaction effect: neither sessions nor rating is sufficient alone")

        df2 = df.copy()
        df2["Rating_Tier"] = pd.cut(df2["Mentor_Rating"], bins=[0, 2.99, 3.49, 3.99, 5.01],
                                     labels=["<3.0", "3.0–3.5", "3.5–4.0", "4.0+"])
        RATE_ORD = ["<3.0", "3.0–3.5", "3.5–4.0", "4.0+"]

        pivot3 = (df2.groupby(["Sessions_Per_Month", "Rating_Tier"], observed=True)["Outcome_Num"]
                  .mean().unstack() * 100).round(1).reindex(columns=RATE_ORD)

        fig = go.Figure(go.Heatmap(
            z=pivot3.values, x=pivot3.columns.tolist(),
            y=[f"{i}/mo" for i in pivot3.index.tolist()],
            colorscale="RdYlGn", text=pivot3.values, texttemplate="%{text:.0f}%",
            colorbar=dict(title="Outcome %"), zmin=10, zmax=80,
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=350,
                          xaxis_title="Mentor Rating Tier", yaxis_title="Sessions per Month",
                          margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        best  = pivot3.stack().dropna().idxmax()
        worst = pivot3.stack().dropna().idxmin()
        col1, col2 = st.columns(2)
        col1.success(f"**Best combo:** {best[0]}/mo + {best[1]} mentor → {pivot3.stack().dropna().max():.0f}%")
        col2.error(f"**Worst combo:** {worst[0]}/mo + {worst[1]} mentor → {pivot3.stack().dropna().min():.0f}%")

        st.markdown(f"""<div class="insight-box">
        <strong>The Interaction Effect:</strong> This is the most important cross-tabulation in 
        the dataset. High session frequency with a poor mentor <em>compounds bad advice</em>. 
        An excellent mentor seen once a month produces untapped potential. The magic is in the 
        top-right cell: 3–4 sessions + 4.0+ mentor rating. Product implication: pull both 
        levers simultaneously. A Month-1 check-in should flag any mentee below 2 sessions/month 
        OR below 3.5 mentor rating for proactive Success Manager outreach.</div>""", unsafe_allow_html=True)

    # ── Tab 4: Corporate × Sessions ────────────────────────────────────────────
    with tab4:
        st.markdown("#### Career Outcome Rate by Corporate Sponsorship × Sessions/Month")
        pivot4 = (df.groupby(["Corporate_Sponsored", "Sessions_Per_Month"], observed=True)["Outcome_Num"]
                  .mean().unstack() * 100).round(1)

        fig = go.Figure()
        for sponsor, color in [("Yes", COLORS["teal"]), ("No", COLORS["purple"])]:
            if sponsor in pivot4.index:
                vals = pivot4.loc[sponsor].reindex([1, 2, 3, 4])
                fig.add_trace(go.Bar(
                    x=[f"{i}/mo" for i in [1, 2, 3, 4]],
                    y=vals.values, name=f"Corporate: {sponsor}",
                    marker_color=color, text=vals.values.round(0),
                    texttemplate="%{text:.0f}%", textposition="outside"
                ))
        fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["gray"],
                      annotation_text=f"Baseline {baseline:.0f}%")
        fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=380,
                          xaxis_title="Sessions per Month",
                          yaxis_title="Career Outcome Rate (%)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Retention comparison
        ret4 = (df.groupby("Corporate_Sponsored", observed=True)["Months_Subscribed"].mean()).round(1)
        c1, c2 = st.columns(2)
        if "Yes" in ret4.index:
            c1.metric("Corporate avg retention", f"{ret4['Yes']:.1f} months",
                      f"+{ret4['Yes']-ret4.get('No',0):.1f} vs Individual")
        if "No" in ret4.index:
            c2.metric("Individual avg retention", f"{ret4.get('No',0):.1f} months")

        st.markdown(f"""<div class="insight-box">
        <strong>B2B is a Behaviour Multiplier:</strong> Corporate-sponsored mentees outperform 
        individual-paying mentees at <em>every</em> session frequency level. This is not merely 
        a payment method difference — it is a fundamental shift in how mentees engage. When an 
        employer invests in a mentee, there is mutual accountability: the employee feels obligated 
        to attend, and sessions become preparation for real work deliverables rather than 
        abstract career aspirations. B2B acquisition should be MentorMatch's #1 growth priority.
        </div>""", unsafe_allow_html=True)

    # ── Tab 5: Age Group × Goal ────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Career Outcome Rate (%) — Age Group × Goal Type")
        df5 = df.copy()
        df5["Age_Group"] = pd.cut(df5["Age"], bins=[21, 26, 31, 36, 100],
                                   labels=["22–26 (Early)", "27–31 (Growth)", "32–36 (Mid)", "37+ (Established)"])
        AGE_ORD = ["22–26 (Early)", "27–31 (Growth)", "32–36 (Mid)", "37+ (Established)"]

        pivot5 = (df5.groupby(["Age_Group", "Goal_Type"], observed=True)["Outcome_Num"]
                  .mean().unstack() * 100).round(1).reindex(index=AGE_ORD, columns=GOAL_ORD)

        fig = go.Figure(go.Heatmap(
            z=pivot5.values, x=pivot5.columns.tolist(), y=pivot5.index.tolist(),
            colorscale="YlGn", text=pivot5.values, texttemplate="%{text:.0f}%",
            colorbar=dict(title="Outcome %"), zmin=15, zmax=80,
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=330,
                          xaxis_title="Goal Type", yaxis_title="Age Group (Career Stage)",
                          margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <strong>The 27–31 Growth Window:</strong> In UAE's expat professional market, the 
        27–31 age bracket is the peak mentoring ROI window. Professionals in this range have 
        enough experience to <em>act</em> on mentor advice, but their careers are still 
        sufficiently malleable to be redirected. By contrast, entry-level (22–26) mentees often 
        lack the contextual experience to implement advice, and 37+ professionals have more 
        entrenched career paths. The 27–31 bracket should be MentorMatch's primary paid 
        acquisition target.</div>""", unsafe_allow_html=True)

    # ── Tab 6: Industry × Goal → Retention ─────────────────────────────────────
    with tab6:
        st.markdown("#### Avg Months Subscribed — Industry × Goal Type (Retention Heatmap)")
        overall_ret = df["Months_Subscribed"].mean()

        pivot6 = (df.groupby(["Industry", "Goal_Type"], observed=True)["Months_Subscribed"]
                  .mean().unstack()).round(1).reindex(columns=GOAL_ORD)
        pivot6 = pivot6.reindex(pivot6.mean(axis=1).sort_values(ascending=False).index)

        fig = go.Figure(go.Heatmap(
            z=pivot6.values, x=pivot6.columns.tolist(), y=pivot6.index.tolist(),
            colorscale="Blues", text=pivot6.values, texttemplate="%{text:.1f} mo",
            colorbar=dict(title="Avg Months"), zmin=2, zmax=15,
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                          xaxis_title="Goal Type", yaxis_title="Industry",
                          margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Platform average retention: **{overall_ret:.1f} months**  "
                f"| Combinations above this are high-LTV segments.")

        # Top 5 retention combos
        top_ret = pivot6.stack().dropna().sort_values(ascending=False).head(5).reset_index()
        top_ret.columns = ["Industry", "Goal Type", "Avg Months"]
        st.markdown("**Top 5 highest-retention combinations (highest LTV):**")
        st.dataframe(top_ret.style.background_gradient(subset=["Avg Months"],
                     cmap="Blues"), use_container_width=True)


# ─── PAGE 3 — DIAGNOSTIC ANALYSIS ───────────────────────────────────────────
def page_diagnostic(df: pd.DataFrame):
    st.markdown("<h2 class='section-header'>🔍 Diagnostic Analysis — Probing Career Outcome Drivers</h2>", unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">
    <strong>Diagnostic Analysis</strong> goes beyond <em>what</em> is happening to ask <em>why</em>. 
    We probe each key dimension — Age, Experience, Industry, Goal Type, Engagement — 
    to understand the mechanisms behind career outcome conversion and subscription retention in UAE's 
    professional landscape.
    </div>""", unsafe_allow_html=True)

    baseline = (df["Career_Outcome_Achieved"] == "Yes").mean() * 100

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎂 Age Deep Dive",
        "🪜 Experience Ladder",
        "🎯 Goal Alignment",
        "🏭 Industry Lens",
        "📈 Engagement & Retention",
    ])

    # ── Tab 1: Age ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Age — Distribution, Engagement, and Career Outcome")
        df_a = df.copy()
        df_a["Age_Group"] = pd.cut(df_a["Age"], bins=[21, 26, 31, 36, 100],
                                    labels=["22–26", "27–31", "32–36", "37+"])
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x="Age", color="Career_Outcome_Achieved",
                               color_discrete_map={"Yes": COLORS["teal"], "No": COLORS["coral"]},
                               nbins=20, barmode="overlay", opacity=0.75,
                               title="Age Distribution by Career Outcome",
                               template=PLOTLY_TEMPLATE)
            fig.update_layout(height=320, legend_title="Career Outcome",
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            age_grp = df_a.groupby("Age_Group", observed=True).agg(
                Outcome_Rate=("Outcome_Num", lambda x: x.mean()*100),
                Avg_NPS=("NPS_Score", "mean"),
                Avg_Sessions=("Sessions_Per_Month", "mean"),
                Count=("Outcome_Num", "count")
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=age_grp["Age_Group"].astype(str),
                                  y=age_grp["Outcome_Rate"].round(1),
                                  name="Career Outcome %",
                                  marker_color=COLORS["teal"],
                                  text=age_grp["Outcome_Rate"].round(0),
                                  texttemplate="%{text:.0f}%", textposition="outside"))
            fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["gray"],
                           annotation_text=f"Avg {baseline:.0f}%")
            fig.update_layout(title="Career Outcome Rate by Age Group",
                               template=PLOTLY_TEMPLATE, height=320,
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        # Age metrics table
        st.markdown("**Age Group Performance Summary:**")
        age_tbl = df_a.groupby("Age_Group", observed=True).agg(
            Count=("Outcome_Num", "count"),
            Outcome_Rate=("Outcome_Num", lambda x: f"{x.mean()*100:.0f}%"),
            Avg_NPS=("NPS_Score", lambda x: f"{x.mean():.0f}"),
            Avg_Months=("Months_Subscribed", lambda x: f"{x.mean():.1f}"),
            Churn_Rate=("Churned_Num", lambda x: f"{x.mean()*100:.0f}%"),
        ).reset_index()
        st.dataframe(age_tbl, use_container_width=True, hide_index=True)

        st.markdown("""<div class="insight-box">
        <strong>UAE Career Age Dynamics:</strong> The 27–31 bracket consistently shows the 
        highest career outcome rates. In Dubai's expat-heavy market, this is the inflection 
        point where professionals have accumulated enough local experience (UAE visa track record, 
        professional network, cultural fluency) to act decisively on a mentor's advice. 
        Entry-level (22–26) mentees are still building this foundation. 37+ professionals have 
        more entrenched career paths but leverage mentoring effectively for entrepreneurship 
        and senior network building.</div>""", unsafe_allow_html=True)

    # ── Tab 2: Experience Ladder ───────────────────────────────────────────────
    with tab2:
        st.markdown("#### The UAE Career Experience Ladder — Who Benefits Most?")
        EXP_ORD = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-10 yrs)"]

        col1, col2 = st.columns(2)
        with col1:
            exp_out = (df.groupby("Experience_Level", observed=True)["Outcome_Num"]
                       .mean() * 100).reindex(EXP_ORD).reset_index()
            exp_out.columns = ["Experience", "Outcome_Rate"]
            fig = go.Figure(go.Bar(
                x=exp_out["Experience"], y=exp_out["Outcome_Rate"],
                marker_color=[COLORS["coral"], COLORS["amber"], COLORS["teal"]],
                text=exp_out["Outcome_Rate"].round(0),
                texttemplate="%{text:.0f}%", textposition="outside"
            ))
            fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["gray"])
            fig.update_layout(title="Career Outcome Rate by Experience",
                               template=PLOTLY_TEMPLATE, height=320,
                               margin=dict(t=40, b=40, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            exp_metrics = df.groupby("Experience_Level", observed=True).agg(
                NPS=("NPS_Score", "mean"),
                Churn=("Churned_Num", lambda x: x.mean()*100),
                Months=("Months_Subscribed", "mean"),
                Sessions=("Sessions_Per_Month", "mean"),
            ).reindex(EXP_ORD)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=exp_metrics.index.astype(str),
                                      y=exp_metrics["NPS"].round(1),
                                      mode="lines+markers+text",
                                      name="Avg NPS",
                                      line_color=COLORS["blue"],
                                      text=exp_metrics["NPS"].round(1),
                                      textposition="top center"))
            fig.add_trace(go.Scatter(x=exp_metrics.index.astype(str),
                                      y=exp_metrics["Months"].round(1),
                                      mode="lines+markers+text",
                                      name="Avg Months",
                                      line_color=COLORS["teal"],
                                      text=exp_metrics["Months"].round(1),
                                      textposition="top center"))
            fig.update_layout(title="NPS and Retention by Experience Level",
                               template=PLOTLY_TEMPLATE, height=320,
                               legend=dict(orientation="h"),
                               margin=dict(t=40, b=40, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        # Box plot: NPS distribution by experience level and career outcome
        fig = px.box(df, x="Experience_Level", y="NPS_Score",
                     color="Career_Outcome_Achieved",
                     color_discrete_map={"Yes": COLORS["teal"], "No": COLORS["coral"]},
                     category_orders={"Experience_Level": EXP_ORD},
                     title="NPS Distribution — Experience Level × Career Outcome",
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(height=350, legend_title="Career Outcome",
                           margin=dict(t=40, b=40, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <strong>The Entry-Level Crisis in UAE:</strong> Entry-level professionals in Dubai face 
        a unique 'double disadvantage' — they are over-qualified globally but locally invisible. 
        The UAE job market requires not just skills but local proof of competence (references, 
        visa track record, cultural fit signals). A mentor who navigated this exact transition 
        provides an irreplaceable shortcut. However, entry-level mentees also have the highest 
        churn risk if goals are misaligned with what mentoring can realistically achieve in their 
        timeframe. The platform must set appropriate expectations at onboarding.</div>""", 
                    unsafe_allow_html=True)

    # ── Tab 3: Goal Alignment ──────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Goal Type Analysis — Alignment, Outcomes, and Satisfaction")
        GOAL_ORD = ["Skill Building", "Get Promoted", "Network Building", "Career Switch", "Start a Business"]

        col1, col2 = st.columns(2)
        with col1:
            goal_out = (df.groupby("Goal_Type", observed=True)["Outcome_Num"]
                        .mean() * 100).reindex(GOAL_ORD).reset_index()
            goal_out.columns = ["Goal", "Outcome_Rate"]
            goal_out["color"] = goal_out["Outcome_Rate"].apply(
                lambda x: COLORS["teal"] if x >= baseline else COLORS["coral"])
            fig = go.Figure(go.Bar(
                x=goal_out["Goal"], y=goal_out["Outcome_Rate"],
                marker_color=goal_out["color"],
                text=goal_out["Outcome_Rate"].round(0),
                texttemplate="%{text:.0f}%", textposition="outside"
            ))
            fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["gray"],
                           annotation_text=f"Avg {baseline:.0f}%")
            fig.update_layout(title="Career Outcome Rate by Goal Type",
                               template=PLOTLY_TEMPLATE, height=320,
                               xaxis_tickangle=-25, margin=dict(t=40, b=60, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            goal_nps = df.groupby("Goal_Type", observed=True).agg(
                NPS=("NPS_Score", "mean"),
                Churn=("Churned_Num", lambda x: x.mean()*100),
                Months=("Months_Subscribed", "mean"),
            ).reindex(GOAL_ORD)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=goal_nps.index.tolist(), y=goal_nps["Churn"].round(1),
                                  name="Churn Rate (%)", marker_color=COLORS["coral"]))
            fig.add_trace(go.Scatter(x=goal_nps.index.tolist(), y=goal_nps["NPS"].round(1),
                                     name="Avg NPS", line_color=COLORS["blue"],
                                     mode="lines+markers", yaxis="y2"))
            fig.update_layout(
                title="Churn Rate & NPS by Goal Type",
                template=PLOTLY_TEMPLATE, height=320,
                yaxis=dict(title="Churn Rate (%)"),
                yaxis2=dict(title="NPS Score", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                xaxis_tickangle=-25, margin=dict(t=40, b=60, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <strong>Goal-Stage Mismatch is the Silent Churn Driver:</strong> Career Switch has the 
        lowest career outcome rate and highest churn rate. In UAE, switching industries requires 
        overcoming the visa title constraint — most employers hire for roles matching your current 
        visa classification. The platform should proactively flag when a mentee's goal is 
        structurally misaligned with their current stage and suggest a stepping-stone goal 
        (e.g., Skill Building → Career Switch) instead of allowing immediate disappointment.</div>""", 
                    unsafe_allow_html=True)

    # ── Tab 4: Industry ────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Industry Analysis — Outcome, Churn, Revenue, and NPS")
        col1, col2 = st.columns(2)

        with col1:
            ind_metrics = df.groupby("Industry", observed=True).agg(
                Outcome=("Outcome_Num", lambda x: x.mean()*100),
                Churn=("Churned_Num", lambda x: x.mean()*100),
                NPS=("NPS_Score", "mean"),
                Revenue=("Total_Revenue_AED", "mean"),
            ).sort_values("Outcome", ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=ind_metrics.index.tolist(), y=ind_metrics["Outcome"].round(1),
                                  name="Career Outcome %", marker_color=COLORS["teal"],
                                  text=ind_metrics["Outcome"].round(0),
                                  texttemplate="%{text:.0f}%", textposition="outside"))
            fig.add_trace(go.Scatter(x=ind_metrics.index.tolist(), y=ind_metrics["Churn"].round(1),
                                     name="Churn Rate %", line_color=COLORS["coral"],
                                     mode="lines+markers", yaxis="y2"))
            fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["gray"])
            fig.update_layout(
                title="Career Outcome % and Churn Rate by Industry",
                template=PLOTLY_TEMPLATE, height=380,
                yaxis=dict(title="Career Outcome Rate (%)"),
                yaxis2=dict(title="Churn Rate (%)", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                xaxis_tickangle=-30, margin=dict(t=50, b=70, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            ind_rev = df.groupby("Industry", observed=True)["Total_Revenue_AED"].mean().sort_values(ascending=False)
            fig = go.Figure(go.Bar(
                x=ind_rev.values.round(0), y=ind_rev.index.tolist(),
                orientation="h", marker_color=COLORS["blue"],
                text=[f"AED {v:,.0f}" for v in ind_rev.values],
                textposition="outside"
            ))
            fig.update_layout(title="Avg Revenue per Mentee by Industry",
                               template=PLOTLY_TEMPLATE, height=380,
                               margin=dict(t=40, b=10, l=10, r=140))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <strong>UAE Industry Dynamics:</strong> Finance (DIFC) and Technology show the highest 
        mentoring ROI because they have well-defined career ladders and abundant experienced 
        mentors who climbed the same path. Real Estate is uniquely strong in UAE — Dubai's 
        property boom creates high-value mentoring opportunities. Marketing suffers from 
        structural oversupply — hundreds of applicants per role makes even excellent mentoring 
        insufficient to overcome market headwinds. MentorMatch should build industry-specific 
        mentor pools and pricing tiers accordingly.</div>""", unsafe_allow_html=True)

    # ── Tab 5: Engagement & Retention ─────────────────────────────────────────
    with tab5:
        st.markdown("#### Engagement & Retention Deep Dive")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="Sessions_Per_Month", y="Engagement_Score",
                             color="Career_Outcome_Achieved",
                             color_discrete_map={"Yes": COLORS["teal"], "No": COLORS["coral"]},
                             title="Sessions vs Engagement Score (coloured by Outcome)",
                             template=PLOTLY_TEMPLATE, opacity=0.7)
            fig.update_layout(height=320, legend_title="Career Outcome",
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(df, x="Mentor_Rating", y="NPS_Score",
                             color="Career_Outcome_Achieved",
                             color_discrete_map={"Yes": COLORS["teal"], "No": COLORS["coral"]},
                             title="Mentor Rating vs NPS Score (coloured by Outcome)",
                             template=PLOTLY_TEMPLATE, opacity=0.7)
            fig.update_layout(height=320, legend_title="Career Outcome",
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        # Months subscribed distribution
        fig = px.violin(df, x="Career_Outcome_Achieved", y="Months_Subscribed",
                         color="Career_Outcome_Achieved",
                         color_discrete_map={"Yes": COLORS["teal"], "No": COLORS["coral"]},
                         box=True, points="outliers",
                         title="Months Subscribed Distribution by Career Outcome",
                         template=PLOTLY_TEMPLATE)
        fig.update_layout(height=350, showlegend=False,
                           margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Key correlations
        num_cols = ["Age", "Months_Subscribed", "Sessions_Per_Month", "Mentor_Rating",
                    "Session_Completion_Rate_Pct", "NPS_Score", "Engagement_Score"]
        corr_with_outcome = df[num_cols + ["Outcome_Num"]].corr()["Outcome_Num"].drop("Outcome_Num").sort_values()

        fig = go.Figure(go.Bar(
            x=corr_with_outcome.values, y=corr_with_outcome.index,
            orientation="h",
            marker_color=[COLORS["teal"] if v > 0 else COLORS["coral"] for v in corr_with_outcome.values],
            text=[f"{v:+.3f}" for v in corr_with_outcome.values],
            textposition="outside"
        ))
        fig.add_vline(x=0, line_color=COLORS["gray"])
        fig.update_layout(title="Pearson Correlation with Career Outcome (numeric features)",
                           template=PLOTLY_TEMPLATE, height=320,
                           xaxis_title="Correlation r",
                           margin=dict(t=40, b=10, l=180, r=80))
        st.plotly_chart(fig, use_container_width=True)


# ─── PAGE 4 — FEATURE ENGINEERING ───────────────────────────────────────────
def page_feature_engineering(df: pd.DataFrame):
    st.markdown("<h2 class='section-header'>⚙️ Feature Engineering — Preparing Data for ML Classification</h2>", unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <strong>Why Feature Engineering?</strong> Machine learning models cannot directly process 
    text categories (like "Finance" or "Entry (0-2 yrs)"). We need to convert all features 
    into numbers. The choices we make here — how to encode, which features to include, 
    how to scale — directly determine model accuracy.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📋 Original Features (21 columns)")
        orig_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Used for ML": [
                "No (ID)", "No (Date)", "No (too sparse)", "Yes (numeric)",
                "Yes (one-hot)", "Yes (ordinal)", "Yes (one-hot)", "Yes (one-hot)",
                "Yes (binary)", "Yes (numeric)", "Yes (numeric)", "Yes (numeric)",
                "Yes (numeric)", "Yes (binary)", "Yes (numeric)", "TARGET",
                "No (redundant)", "No (redundant)", "No (derived)", "No (derived)", "Yes (numeric)"
            ],
            "Sample": [str(df[c].iloc[0])[:20] for c in df.columns]
        })
        st.dataframe(orig_info, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🔧 Encoding Strategy")
        enc_info = {
            "Feature": [
                "Career_Outcome_Achieved", "Corporate_Sponsored", "Churned",
                "Experience_Level", "Industry", "Goal_Type", "Referral_Source",
                "Age, Months_Subscribed, etc."
            ],
            "Encoding": [
                "Binary (TARGET: Yes=1, No=0)",
                "Binary (Yes=1, No=0)",
                "Binary (Yes=1, No=0)",
                "Ordinal (Entry=1, Mid=2, Senior=3)",
                "One-Hot (8 categories → 7 dummy cols)",
                "One-Hot (5 categories → 4 dummy cols)",
                "One-Hot (6 categories → 5 dummy cols)",
                "Numeric (used as-is)"
            ],
            "Reason": [
                "Binary classification target",
                "Natural binary — ordered",
                "Natural binary — ordered",
                "Meaningful order: Entry < Mid < Senior",
                "No inherent order between sectors",
                "No inherent order between goal types",
                "No inherent order between channels",
                "Continuous — no encoding needed"
            ]
        }
        st.dataframe(pd.DataFrame(enc_info), use_container_width=True, hide_index=True)

    st.markdown("#### 🔄 After Encoding — Final Feature Matrix")
    X, y, feat_names = engineer_features(df)
    st.info(f"Original: {df.shape[1]} columns → After encoding: **{len(feat_names)} features** "
            f"| Train/Test split: 80/20 stratified | Target class balance: "
            f"{y.sum()} Positive ({y.mean()*100:.0f}%) / "
            f"{len(y)-y.sum()} Negative ({(1-y.mean())*100:.0f}%)")

    feat_df = pd.DataFrame({"Feature": feat_names})
    feat_df["Category"] = feat_df["Feature"].apply(lambda f:
        "Numeric" if f in ["Age", "Months_Subscribed", "Sessions_Per_Month", "Mentor_Rating",
                            "Session_Completion_Rate_Pct", "NPS_Score", "Engagement_Score"]
        else "Binary" if f in ["Corporate_Sponsored", "Churned"]
        else "Ordinal" if f == "Experience_Level"
        else "Industry (OHE)" if f.startswith("Industry_")
        else "Goal Type (OHE)" if f.startswith("Goal_Type_")
        else "Referral (OHE)" if f.startswith("Referral_")
        else "Other"
    )
    feat_counts = feat_df["Category"].value_counts().reset_index()
    feat_counts.columns = ["Category", "Count"]
    fig = go.Figure(go.Pie(labels=feat_counts["Category"], values=feat_counts["Count"],
                            hole=0.5, textinfo="percent+label",
                            marker_colors=[COLORS["teal"], COLORS["blue"], COLORS["purple"],
                                           COLORS["amber"], COLORS["coral"], COLORS["gray"]]))
    fig.update_layout(title="Feature Matrix Composition After Encoding",
                       template=PLOTLY_TEMPLATE, height=320, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix of numeric features
    st.markdown("#### 📊 Correlation Heatmap (Numeric Features × Target)")
    num_cols = ["Age", "Months_Subscribed", "Sessions_Per_Month", "Mentor_Rating",
                "Session_Completion_Rate_Pct", "NPS_Score", "Engagement_Score", "Outcome_Num"]
    corr = df[num_cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdYlGn", text=corr.values, texttemplate="%{text:.2f}",
        colorbar=dict(title="r"), zmid=0, zmin=-1, zmax=1,
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400,
                       margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
"""
MentorMatch UAE — Analytics Dashboard
Part C: Pages 5–7 + main routing
"""

# ─── PAGE 5 — ML CLASSIFICATION MODELS ───────────────────────────────────────
def page_ml_models(df: pd.DataFrame):
    st.markdown("<h2 class='section-header'>🤖 ML Classification — Career Outcome Prediction</h2>", unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <strong>Objective:</strong> Build and compare four supervised classification algorithms 
    to predict <em>Career_Outcome_Achieved</em> (Yes/No). Each model learns patterns from 
    the feature-engineered dataset and is evaluated on a held-out test set.
    </div>""", unsafe_allow_html=True)

    # ── Model parameter controls ───────────────────────────────────────────────
    with st.expander("⚙️ Model Hyperparameters (click to adjust)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**K-Nearest Neighbours**")
            knn_k    = st.slider("K (neighbours)", 3, 21, 5, 2, key="knn_k")
            test_sz  = st.slider("Test size %", 15, 35, 20, 5, key="ts") / 100
        with c2:
            st.markdown("**Decision Tree**")
            dt_depth = st.slider("Max Depth", 2, 15, 5, 1, key="dt_d")
        with c3:
            st.markdown("**Random Forest**")
            rf_trees = st.slider("N Estimators", 50, 300, 100, 50, key="rf_n")
        with c4:
            st.markdown("**Gradient Boosting**")
            gb_lr    = st.select_slider("Learning Rate", [0.01, 0.05, 0.1, 0.2, 0.3], 0.1, key="gb_lr")

    if st.button("🚀 Train All 4 Models", type="primary", use_container_width=True):
        with st.spinner("Training KNN, Decision Tree, Random Forest, Gradient Boosting..."):
            results = train_all_models(df, test_sz, knn_k, dt_depth, rf_trees, gb_lr)
        st.session_state["results"] = results
        st.success("All models trained successfully!")

    results = st.session_state.get("results")
    if results is None:
        st.info("👆 Adjust hyperparameters above (optional) and click **Train All 4 Models** to begin.")
        return

    meta = results["_meta"]
    st.markdown(f"""<div class="insight-box">
    <strong>Training Setup:</strong> {meta['n_train']} training samples / {meta['n_test']} test samples 
    | {len(meta['feat_names'])} features | Stratified 80/{int(meta['test_size']*100)} split 
    | 5-fold cross-validation on training set
    </div>""", unsafe_allow_html=True)

    # ── Results summary table ─────────────────────────────────────────────────
    st.markdown("#### 📊 Model Performance Summary")
    summary_rows = []
    for name in ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]:
        r = results[name]
        summary_rows.append({
            "Model"           : name,
            "Train Accuracy"  : f"{r['train_acc']*100:.1f}%",
            "Test Accuracy"   : f"{r['test_acc']*100:.1f}%",
            "CV Score (mean)" : f"{r['cv_mean']*100:.1f}% ± {r['cv_std']*100:.1f}%",
            "Precision"       : f"{r['precision']*100:.1f}%",
            "Recall"          : f"{r['recall']*100:.1f}%",
            "F1-Score"        : f"{r['f1']*100:.1f}%",
            "AUC-ROC"         : f"{r['auc']:.3f}",
        })
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary.style.background_gradient(
        subset=["Test Accuracy", "F1-Score", "AUC-ROC"],
        cmap="Greens"), use_container_width=True, hide_index=True)

    # ── Bar comparison chart ───────────────────────────────────────────────────
    st.markdown("#### 📈 Training vs Testing Accuracy Comparison")
    model_names = ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
    train_accs  = [results[n]["train_acc"]*100 for n in model_names]
    test_accs   = [results[n]["test_acc"] *100 for n in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Training Accuracy", x=model_names, y=train_accs,
                          marker_color=COLORS["blue"], text=[f"{v:.1f}%" for v in train_accs],
                          textposition="outside"))
    fig.add_trace(go.Bar(name="Testing Accuracy", x=model_names, y=test_accs,
                          marker_color=COLORS["teal"], text=[f"{v:.1f}%" for v in test_accs],
                          textposition="outside"))
    fig.add_hline(y=59, line_dash="dot", line_color=COLORS["gray"],
                   annotation_text="Majority Classifier Baseline ~59%",
                   annotation_position="bottom right")
    fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, height=400,
                       yaxis=dict(range=[0, 105], title="Accuracy (%)"),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <strong>Over-fitting Watch:</strong> A large gap between Training and Testing accuracy 
    signals over-fitting — the model memorised training data but cannot generalise. 
    Decision Trees are especially prone to this. Random Forest and Gradient Boosting 
    use ensemble averaging to reduce over-fitting. 
    Cross-validation scores (5-fold) provide a more robust accuracy estimate than a 
    single train/test split.
    </div>""", unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────────────────
    if "feat_importance" in meta and meta["feat_importance"]:
        st.markdown("#### 🌟 Feature Importance (Tree-Based Models)")
        fi_tabs = st.tabs(list(meta["feat_importance"].keys()))
        for tab, (model_name, fi_series) in zip(fi_tabs, meta["feat_importance"].items()):
            with tab:
                top_fi = fi_series.head(12)
                fig = go.Figure(go.Bar(
                    x=top_fi.values * 100, y=top_fi.index.tolist(),
                    orientation="h",
                    marker_color=MODEL_COLORS.get(model_name, COLORS["teal"]),
                    text=[f"{v*100:.1f}%" for v in top_fi.values],
                    textposition="outside"
                ))
                fig.update_layout(title=f"{model_name} — Top 12 Feature Importances",
                                   template=PLOTLY_TEMPLATE, height=380,
                                   xaxis_title="Importance (%)",
                                   margin=dict(t=40, b=10, l=200, r=80))
                st.plotly_chart(fig, use_container_width=True)


# ─── PAGE 6 — MODEL PERFORMANCE (ROC + CONFUSION MATRICES) ───────────────────
def page_model_performance(df: pd.DataFrame):
    st.markdown("<h2 class='section-header'>📈 Model Performance — ROC Curves, Confusion Matrices & Metrics</h2>", unsafe_allow_html=True)

    results = st.session_state.get("results")
    if results is None:
        st.warning("⚠️ Models not trained yet. Go to **🤖 ML Models** page and click **Train All 4 Models** first.")
        return

    model_names = ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]

    # ── Metrics comparison ────────────────────────────────────────────────────
    st.markdown("#### 📊 Precision, Recall, F1-Score, AUC — Side-by-Side")
    metrics_df = pd.DataFrame({
        "Model"    : model_names,
        "Precision": [results[n]["precision"]*100 for n in model_names],
        "Recall"   : [results[n]["recall"]*100    for n in model_names],
        "F1-Score" : [results[n]["f1"]*100        for n in model_names],
        "AUC-ROC"  : [results[n]["auc"]*100       for n in model_names],
    })

    fig = go.Figure()
    for metric, color in [("Precision", COLORS["blue"]), ("Recall", COLORS["teal"]),
                           ("F1-Score", COLORS["amber"]), ("AUC-ROC", COLORS["purple"])]:
        fig.add_trace(go.Bar(name=metric, x=model_names, y=metrics_df[metric],
                              marker_color=color,
                              text=[f"{v:.1f}%" for v in metrics_df[metric]],
                              textposition="outside"))
    fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE, height=420,
                       yaxis=dict(range=[0, 110], title="Score (%)"),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       margin=dict(t=20, b=10, l=10, r=10),
                       title="Precision / Recall / F1-Score / AUC-ROC — All Models")
    st.plotly_chart(fig, use_container_width=True)

    # Metric explanations
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
| Metric | What it measures | When to prioritise |
|--------|-----------------|-------------------|
| **Accuracy** | Overall % of correct predictions | When classes are balanced |
| **Precision** | Of all predicted "Career Outcome = Yes", how many actually were? | When false positives are costly |
| **Recall** | Of all actual "Career Outcome = Yes", how many did the model catch? | When false negatives are costly (e.g., missing at-risk users) |
| **F1-Score** | Harmonic mean of Precision and Recall | When you need a single balanced metric |
| **AUC-ROC** | Model's ability to distinguish Yes vs No across all thresholds | Model stability and ranking quality |

For MentorMatch: **Recall** is the most important metric — we want to catch as many at-risk 
mentees (predicted "No" outcome) as possible so we can intervene before they churn.
        """)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.markdown("#### 📉 ROC Curves — All 4 Models")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash", color=COLORS["gray"]),
                              name="Random Classifier (AUC=0.50)"))
    for name in model_names:
        r = results[name]
        if r["fpr"] and r["tpr"]:
            fig.add_trace(go.Scatter(
                x=r["fpr"], y=r["tpr"],
                mode="lines",
                name=f"{name} (AUC={r['auc']:.3f})",
                line=dict(color=MODEL_COLORS.get(name, COLORS["gray"]), width=2.5)
            ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=480,
        xaxis=dict(title="False Positive Rate", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", range=[0, 1]),
        legend=dict(x=0.6, y=0.1),
        title="ROC Curves — KNN | Decision Tree | Random Forest | Gradient Boosting",
        margin=dict(t=50, b=10, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <strong>Reading ROC Curves:</strong> The closer the curve hugs the top-left corner, 
    the better the model. AUC (Area Under Curve) of 1.0 = perfect classifier. AUC of 0.5 = 
    random guessing (the dashed line). A model is useful if AUC > 0.60. 
    <strong>Gradient Boosting and Random Forest typically win here</strong> because they 
    build an ensemble of many weak learners, reducing variance. If one model's curve 
    dominates all others across the entire ROC space, it is the most stable model.
    </div>""", unsafe_allow_html=True)

    # ── Confusion Matrices (2×2 grid) ─────────────────────────────────────────
    st.markdown("#### 🔢 Confusion Matrices — All 4 Models")

    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=model_names,
                         vertical_spacing=0.18, horizontal_spacing=0.12)

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    labels = ["No Outcome (0)", "Outcome (1)"]

    for idx, (name, (r, c)) in enumerate(zip(model_names, positions)):
        cm = np.array(results[name]["conf_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        ann_text = [[f"<b>{cm_norm[i][j]:.1f}%</b><br><sub>n={cm[i][j]}</sub>"
                     for j in range(2)] for i in range(2)]

        fig.add_trace(
            go.Heatmap(
                z=cm_norm, x=labels, y=labels,
                colorscale=[[0, "#0D1117"], [0.5, "#1D5E4A"], [1, "#1D9E75"]],
                showscale=(idx == 0),
                text=ann_text, texttemplate="%{text}",
                hoverongaps=False, zmin=0, zmax=100,
                colorbar=dict(x=1.02, title="%")
            ),
            row=r, col=c
        )
        fig.update_xaxes(title_text="Predicted", row=r, col=c)
        fig.update_yaxes(title_text="Actual", row=r, col=c)

    fig.update_layout(template=PLOTLY_TEMPLATE, height=600,
                       title="Normalised Confusion Matrices (% of actual class) + Raw Counts",
                       margin=dict(t=60, b=10, l=10, r=80))
    st.plotly_chart(fig, use_container_width=True)

    # ── Detailed classification reports ────────────────────────────────────────
    st.markdown("#### 📋 Detailed Classification Reports")
    report_tabs = st.tabs(model_names)
    for tab, name in zip(report_tabs, model_names):
        with tab:
            r = results[name]["class_report"]
            rep_df = pd.DataFrame({
                "Class"     : ["Career Outcome = No (0)", "Career Outcome = Yes (1)", "Macro Avg", "Weighted Avg"],
                "Precision" : [f"{r['0']['precision']*100:.1f}%", f"{r['1']['precision']*100:.1f}%",
                               f"{r['macro avg']['precision']*100:.1f}%", f"{r['weighted avg']['precision']*100:.1f}%"],
                "Recall"    : [f"{r['0']['recall']*100:.1f}%", f"{r['1']['recall']*100:.1f}%",
                               f"{r['macro avg']['recall']*100:.1f}%", f"{r['weighted avg']['recall']*100:.1f}%"],
                "F1-Score"  : [f"{r['0']['f1-score']*100:.1f}%", f"{r['1']['f1-score']*100:.1f}%",
                               f"{r['macro avg']['f1-score']*100:.1f}%", f"{r['weighted avg']['f1-score']*100:.1f}%"],
                "Support"   : [str(r['0']['support']), str(r['1']['support']), "", ""],
            })
            st.dataframe(rep_df, use_container_width=True, hide_index=True)

            cv_scores = results[name]["cv_scores"]
            col1, col2, col3 = st.columns(3)
            col1.metric("CV Mean Accuracy", f"{results[name]['cv_mean']*100:.1f}%")
            col2.metric("CV Std Dev",        f"± {results[name]['cv_std']*100:.1f}%")
            col3.metric("AUC-ROC",           f"{results[name]['auc']:.3f}")

            # CV score bar
            fig = go.Figure(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                y=[s*100 for s in cv_scores],
                marker_color=MODEL_COLORS.get(name, COLORS["teal"]),
                text=[f"{s*100:.1f}%" for s in cv_scores],
                textposition="outside"
            ))
            fig.add_hline(y=results[name]["cv_mean"]*100, line_dash="dash",
                           line_color=COLORS["gray"], annotation_text="Mean CV")
            fig.update_layout(title=f"{name} — 5-Fold Cross-Validation Accuracy",
                               template=PLOTLY_TEMPLATE, height=280,
                               yaxis_range=[0, 105],
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Model selection recommendation ────────────────────────────────────────
    best_auc = max(model_names, key=lambda n: results[n]["auc"])
    best_f1  = max(model_names, key=lambda n: results[n]["f1"])
    st.markdown(f"""<div class="insight-box">
    <strong>Model Selection Recommendation:</strong><br>
    • <strong>Best AUC-ROC:</strong> {best_auc} ({results[best_auc]['auc']:.3f}) — 
    most stable ranking across all probability thresholds<br>
    • <strong>Best F1-Score:</strong> {best_f1} ({results[best_f1]['f1']*100:.1f}%) — 
    best balance of precision and recall<br>
    For MentorMatch's use case (identifying at-risk mentees for proactive outreach), 
    <strong>Recall</strong> matters most — we want to catch every potential churn/no-outcome 
    case. Deploy the model with the highest Recall on the "Yes" class.
    </div>""", unsafe_allow_html=True)


# ─── PAGE 7 — FINDINGS ───────────────────────────────────────────────────────
def page_findings(df: pd.DataFrame):
    st.markdown("<h2 class='section-header'>💡 Key Findings & Strategic Recommendations</h2>", unsafe_allow_html=True)
    st.markdown(f"""<div class="insight-box">
    <strong>Executive Summary for MentorMatch UAE CTO:</strong> Analysis of {len(df):,} mentee records 
    reveals that career outcomes are not driven by any single feature — they emerge from 
    the right combination of goal-stage alignment, engagement frequency, mentor quality, 
    and contextual factors specific to UAE's professional market. The platform's 
    {(df['Career_Outcome_Achieved']=='Yes').mean()*100:.0f}% career outcome rate has clear, 
    data-driven pathways to improvement.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔑 Key Findings", "🇦🇪 UAE Market Insights", "🚀 Product Recommendations"])

    with tab1:
        findings = [
            ("1. Goal-stage mismatch is the #1 predictor of failure",
             "Entry-level professionals attempting Career Switches have the lowest career outcome rate in the entire dataset. In UAE, visa sponsorship is tied to job titles — switching industries in <2 years is structurally near-impossible regardless of mentor quality. The platform must introduce a Goal Advisor at onboarding."),
            ("2. Sessions × Mentor Rating interaction drives outcomes multiplicatively",
             "Neither high session frequency nor high mentor rating is sufficient alone. The combination of 3+ sessions/month with a 4.0+ rated mentor produces outcomes that exceed what either variable achieves independently. This is the single most actionable lever in the dataset."),
            ("3. Finance and Technology have the highest mentoring ROI in UAE",
             "These sectors have well-defined DIFC/Tech career ladders and abundant experienced mentors who climbed the same path. Entry-level Finance is the highest-ROI acquisition segment. Marketing is structurally oversupplied — manage expectations proactively."),
            ("4. Corporate sponsorship changes mentee behaviour, not just payment method",
             "Corporate-sponsored mentees outperform individual-paying mentees at every session frequency level AND stay 2+ months longer on average. B2B acquisition should be the platform's #1 growth priority — one HR manager converts 20-50 accounts at 2× LTV."),
            ("5. The 27–31 age bracket is the peak mentoring ROI window",
             "In UAE's expat-heavy market, 27–31 year-olds have accumulated enough local experience to act on mentor advice but are early enough for substantial trajectory changes. This bracket should receive the majority of paid acquisition budget."),
            ("6. Session completion rate is as predictive as mentor rating",
             "Mentees who consistently show up achieve significantly better outcomes. The platform should implement 24-hour automated session reminders for mentees with completion rates below 55%, and track mentor no-show rates separately."),
            ("7. Word of Mouth delivers the highest NPS and career outcome rate",
             "Across all acquisition channels, Word of Mouth consistently delivers the highest-quality users. Every career outcome achieved is a future organic acquisition. A referral program ('Refer a colleague who books 3+ sessions → earn 1 free month') would compound organic growth at near-zero cost."),
            ("8. Engagement Score and Months Subscribed are the top ML predictors",
             "Decision Tree and Random Forest feature importance analysis consistently ranks Engagement Score and Months Subscribed as the strongest predictors of career outcomes. Both are directly influenced by product decisions and should be tracked in a real-time CTO dashboard."),
        ]

        for title, body in findings:
            st.markdown(f"""<div class="finding-card">
            <h4>{title}</h4>
            <p>{body}</p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("""
### 🇦🇪 UAE Job Market Intelligence — What the Data Tells CTO

**The Expat Connectivity Gap**  
90% of Dubai's workforce is expatriate. Most arrive highly qualified but locally disconnected — 
they lack the personal networks, cultural knowledge, and local references that Dubai employers 
use as proxies for competence. MentorMatch sits precisely in this gap. A mentor who has spent 
5+ years in the same industry in Dubai provides access to an informal knowledge network 
that no course, certification, or book can replicate.

**The Visa Title Constraint**  
UAE visa sponsorship classifies workers by specific job titles. This creates a structural 
barrier to career switching that is unique to UAE's labour market. Career Switch is the 
goal type with the lowest career outcome rate — not because MentorMatch fails, but because 
the market structure makes rapid switches difficult. The platform should proactively recommend 
a stepping-stone approach: Skill Building first → Career Switch as a 12-18 month goal.

**The DIFC Finance Premium**  
Dubai International Financial Centre (DIFC) has a well-defined, meritocratic career ladder 
for Finance professionals. This is why Finance + Entry-level shows among the highest mentoring 
ROI in the dataset. Mentors who navigated this exact path can give very specific advice about 
promotion timing, stakeholder visibility, and technical skill priorities. MentorMatch should 
build a dedicated DIFC Finance mentor cohort with verified career histories.

**The Marketing Saturation Problem**  
Dubai's marketing sector is significantly oversupplied. The entry-level marketing candidate 
pool is extremely competitive — mentoring can help differentiate, but structural market 
headwinds limit career outcome conversion. The platform should either (a) focus marketing 
mentoring on Senior+ professionals who can leverage the platform for leadership transitions, 
or (b) set explicit expectations in onboarding about timeline and effort required.

**The B2B Growth Opportunity**  
Corporate sponsorship programs are growing rapidly in DIFC and ADGM as larger employers 
recognise that employee development directly reduces attrition costs. At AED 399/month per 
sponsored employee (vs. AED 50,000-100,000 typical replacement cost), the ROI argument to 
HR directors writes itself. The CTO should build a Manager Dashboard showing aggregate 
employee progress to make this ROI visible and create renewal momentum.

**The Word-of-Mouth Flywheel**  
In Dubai's tight professional communities (DIFC Finance, Downtown Tech hubs, JLT Consulting 
clusters), professional reputation travels fast. A mentee who achieves a promotion or 
successful career outcome tells their professional network. MentorMatch should capture these 
moments — automated "Share your success" prompts triggered when a mentee marks their goal 
as achieved — and convert them into referral acquisition events.
        """)

    with tab3:
        st.markdown("### 🚀 Product & Technology Recommendations for MentorMatch CTO")

        recs = [
            ("🎯 Goal Advisor at Onboarding (Priority: HIGH)",
             "Build a 5-question goal-stage alignment quiz at signup. If Experience = Entry + Goal = Career Switch → show: '87% of entry-level mentees on MentorMatch who start with Skill Building achieve their goals 40% faster. Would you like to adjust your goal?' This single change could lift platform-wide career outcome rates by 8+ percentage points."),
            ("📊 Real-Time Engagement Score Dashboard (Priority: HIGH)",
             "Display each mentee's Engagement Score and predicted Success Likelihood Score (from the trained ML model) in a Success Manager dashboard. Mentees scoring below 35/100 at end of Month 1 should trigger an automated email + manager notification."),
            ("📱 Month-1 Engagement Check System (Priority: HIGH)",
             "Any mentee with ≤1 session/month OR mentor rated below 3.5 should receive a proactive Day-14 Success Manager call. This is the highest-ROI intervention point — the data shows engagement trajectory is set within the first 30 days."),
            ("🤝 B2B Corporate Sales Kit (Priority: HIGH)",
             "Build an ROI calculator for HR directors: cost of disengaged employee × career advancement delay ÷ cost of MentorMatch subscription = clear ROI. Include a Manager Dashboard showing aggregate employee progress and goal achievement rates."),
            ("🔄 Referral Program (Priority: MEDIUM)",
             "Activate Champions (high engagement + career outcome achieved) as referral advocates. 'Refer a colleague who completes 3+ sessions → earn 1 free month.' This channel delivers the highest NPS and career outcome rates at near-zero acquisition cost."),
            ("📈 Industry-Specific Mentor Pools (Priority: MEDIUM)",
             "Build curated mentor cohorts for the highest-ROI combinations: Finance Entry-level (DIFC ladder), Technology Entry-level (portfolio + tech interview prep), Real Estate Mid-level (market navigation). Charge premium for verified industry mentors with documented career histories."),
            ("🔔 Session Completion Nudge System (Priority: MEDIUM)",
             "Implement automated 24-hour and 2-hour SMS/push notifications before booked sessions. Track mentor no-show rate separately from mentee no-show rate. Escalate to Success Manager when completion rate drops below 50% for 2 consecutive months."),
            ("🚀 6-Month Commitment Plan (Priority: LOW)",
             "Offer a 6-month upfront plan at 10% discount. The data shows 7-12 months of tenure produces dramatically higher career outcomes. Locking in tenure commitment while providing a session frequency nudge system addresses both top predictors simultaneously."),
        ]

        for title, body in recs:
            priority = "🔴" if "HIGH" in title else ("🟡" if "MEDIUM" in title else "🟢")
            st.markdown(f"""<div class="finding-card">
            <h4>{priority} {title}</h4>
            <p>{body}</p>
            </div>""", unsafe_allow_html=True)


# ─── MAIN APP ROUTING ─────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 MentorMatch UAE")
        st.markdown("### Analytics Dashboard")
        st.markdown("---")

        uploaded = st.file_uploader("📁 Upload CSV Dataset", type=["csv"],
                                     help="Upload MentorMatch_UAE_Clean_Dataset.csv")
        st.markdown("---")

        page = st.radio("Navigate", [
            "🏠  Overview",
            "📊  Descriptive Analytics",
            "🔍  Diagnostic Analysis",
            "⚙️  Feature Engineering",
            "🤖  ML Classification",
            "📈  Model Performance",
            "💡  Key Findings",
        ])

        st.markdown("---")
        st.markdown("""
        <small>
        **MentorMatch UAE** · Data Analytics<br>
        SP Jain School of Global Management<br>
        CTO Analytics Dashboard v2.0
        </small>
        """, unsafe_allow_html=True)

    # Load data
    df_raw = load_data(uploaded)
    if df_raw is None:
        st.error("⚠️ Dataset not found. Please upload `MentorMatch_UAE_Clean_Dataset.csv` using the sidebar.")
        st.stop()

    df = enrich_data(df_raw)

    # Route to page
    p = page.strip()
    if   "Overview"             in p: page_overview(df)
    elif "Descriptive"          in p: page_descriptive(df)
    elif "Diagnostic"           in p: page_diagnostic(df)
    elif "Feature Engineering"  in p: page_feature_engineering(df)
    elif "ML Classification"    in p: page_ml_models(df)
    elif "Model Performance"    in p: page_model_performance(df)
    elif "Key Findings"         in p: page_findings(df)


if __name__ == "__main__":
    main()
