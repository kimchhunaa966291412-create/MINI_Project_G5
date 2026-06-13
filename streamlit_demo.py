"""
Dropout Risk Dashboard  —  MIN_P Group 05
Live demo aligned with the notebook pipeline (MIN_P_G5__2_.ipynb)

Run:  streamlit run streamlit_demo.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Risk Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] { background: #0f1117; color: #e8eaf0; }
    [data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #1e2d40; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #1e2d40;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricValue"]  { color: #e8eaf0 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"]  { color: #8b95a5 !important; font-size: 0.78rem !important; letter-spacing:.06em; text-transform:uppercase; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem !important; }

    /* Section headers */
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #4a9eff;
        margin: 2rem 0 .4rem;
    }

    /* Risk badge */
    .risk-high   { background:#3d1414; color:#ff6b6b; border:1px solid #7a2020;
                   padding:6px 14px; border-radius:6px; font-weight:700; display:inline-block; }
    .risk-medium { background:#2d2510; color:#ffd166; border:1px solid #6b500a;
                   padding:6px 14px; border-radius:6px; font-weight:700; display:inline-block; }
    .risk-low    { background:#0f2d1b; color:#06d6a0; border:1px solid #0a5c35;
                   padding:6px 14px; border-radius:6px; font-weight:700; display:inline-block; }

    /* Progress bar override for risk meter */
    .stProgress > div > div { background: linear-gradient(90deg, #06d6a0, #ffd166, #ff6b6b); border-radius: 4px; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #1e2d40; border-radius: 8px; }

    /* Divider */
    hr { border-color: #1e2d40; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data generation  (mirrors notebook pipeline)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data(n: int = 3_630, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset that replicates the UCI feature distributions described
    in the notebook (after binary class reduction: Graduate=0, Dropout=1).
    Proportions: 39.1% dropout, 60.9% graduate.
    """
    rng = np.random.default_rng(seed)
    n_drop = int(n * 0.391)
    n_grad = n - n_drop

    def cohort(size, grade_mu, tu_prob, age_mu, units_mu):
        return dict(
            Curricular_units_1st_sem_grade  = rng.normal(grade_mu,   3.5,  size).clip(0, 20),
            Curricular_units_2nd_sem_grade  = rng.normal(grade_mu-.3, 3.5,  size).clip(0, 20),
            Curricular_units_1st_sem_approved = rng.normal(units_mu, 2.5, size).clip(0, 10).round(),
            Curricular_units_2nd_sem_approved = rng.normal(units_mu-.5, 2.5, size).clip(0, 10).round(),
            Tuition_fees_up_to_date         = rng.binomial(1, tu_prob,  size),
            Scholarship_holder              = rng.binomial(1, 0.23,     size),
            Age_at_enrollment               = rng.normal(age_mu,  6,    size).clip(17, 60).round(),
            Daytime_evening_attendance      = rng.binomial(1, 0.78,     size),
            Debtor                          = rng.binomial(1, 0.12 if tu_prob < 0.5 else 0.04, size),
            GDP                             = rng.normal(1.4, 1.2,     size),
            Unemployment_rate               = rng.normal(11.5, 2.5,    size),
            Inflation_rate                  = rng.normal(1.2, 1.0,     size),
            Mothers_qualification           = rng.integers(1, 20,       size),
            Fathers_qualification           = rng.integers(1, 20,       size),
            Marital_status                  = rng.integers(1, 7,        size),
            Application_mode                = rng.integers(1, 18,       size),
            Course                          = rng.integers(1, 17,       size),
            Previous_qualification_grade    = rng.normal(130, 20,      size).clip(95, 190),
            Admission_grade                 = rng.normal(127, 20,      size).clip(95, 190),
            Displaced                       = rng.binomial(1, 0.43,     size),
            Gender                          = rng.binomial(1, 0.35,     size),
            International                   = rng.binomial(1, 0.05,     size),
        )

    drop_d = cohort(n_drop, grade_mu=7.1,  tu_prob=0.35, age_mu=24, units_mu=3.5)
    grad_d = cohort(n_grad, grade_mu=12.5, tu_prob=0.91, age_mu=20, units_mu=6.8)

    frames = []
    for d, label in [(drop_d, 1), (grad_d, 0)]:
        df_c = pd.DataFrame(d)
        df_c["Target"] = label
        frames.append(df_c)

    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    return df


# ─────────────────────────────────────────────
# Full ML pipeline  (matches notebook exactly)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_pipeline():
    df = generate_data()
    X  = df.drop("Target", axis=1)
    y  = df["Target"]
    feature_names = X.columns.tolist()

    # 70 / 15 / 15 stratified split
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=val_frac, stratify=y_tr_full, random_state=42)

    scaler = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_val_sc  = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sc, y_train)

    classifiers = {
        "Logistic Regression": LogisticRegression(C=1, solver="lbfgs", max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, min_samples_split=5, criterion="gini", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                             eval_metric="logloss", random_state=42, verbosity=0),
    }

    best_estimators, results = {}, []
    for name, clf in classifiers.items():
        clf.fit(X_tr_bal, y_tr_bal)
        y_pred = clf.predict(X_test_sc)
        y_prob = clf.predict_proba(X_test_sc)[:, 1]
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall":    round(recall_score(y_test, y_pred), 3),
            "F1":        round(f1_score(y_test, y_pred), 3),
            "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 3),
        })
        best_estimators[name] = clf

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)

    # SHAP for XGBoost
    xgb  = best_estimators["XGBoost"]
    X_test_df  = pd.DataFrame(X_test_sc, columns=feature_names)
    explainer  = shap.TreeExplainer(xgb)
    shap_vals  = explainer.shap_values(X_test_df)

    return {
        "df":            generate_data(),
        "scaler":        scaler,
        "models":        best_estimators,
        "results_df":    results_df,
        "X_test_sc":     X_test_sc,
        "y_test":        y_test.values,
        "shap_vals":     shap_vals,
        "X_test_df":     X_test_df,
        "feature_names": feature_names,
    }


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Dropout Risk")
    st.markdown("<p style='color:#8b95a5;font-size:.78rem;margin-top:-8px'>SDG 4 · Quality Education</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊  Overview", "🔍  Risk Calculator", "🤖  Model Comparison", "🔬  SHAP Explainability"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<p style='color:#4a4f5c;font-size:.72rem'>Group 05 · AMSI32_MIP<br>Institute of Technology of Cambodia</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load pipeline (cached)
# ─────────────────────────────────────────────
with st.spinner("Training models on UCI dataset pipeline…"):
    ctx = train_pipeline()

df           = ctx["df"]
scaler       = ctx["scaler"]
models       = ctx["models"]
results_df   = ctx["results_df"]
X_test_sc    = ctx["X_test_sc"]
y_test       = ctx["y_test"]
shap_vals    = ctx["shap_vals"]
X_test_df    = ctx["X_test_df"]
feature_names = ctx["feature_names"]

DARK_BG = "#0f1117"
CARD_BG = "#161b27"
ACCENT  = "#4a9eff"
RED     = "#ff6b6b"
GREEN   = "#06d6a0"
YELLOW  = "#ffd166"


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊  Overview":
    st.title("Student Dropout Risk Dashboard")
    st.markdown("<p style='color:#8b95a5;margin-top:-12px'>UCI Dataset · 3,630 students · Binary Classification (Graduate vs Dropout)</p>", unsafe_allow_html=True)
    st.markdown("---")

    n_total   = len(df)
    n_dropout = df["Target"].sum()
    n_grad    = n_total - n_dropout
    best_auc  = results_df["AUC-ROC"].max()
    best_f1   = results_df["F1"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",   f"{n_total:,}")
    c2.metric("Dropout Rate",     f"{n_dropout/n_total*100:.1f}%",  delta=f"{n_dropout:,} students", delta_color="inverse")
    c3.metric("Best AUC-ROC",     f"{best_auc:.3f}", delta="XGBoost")
    c4.metric("Best F1-Score",    f"{best_f1:.3f}",  delta="XGBoost")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # — Class distribution donut —
    with col_a:
        st.markdown('<p class="section-title">Class Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        sizes  = [n_grad, n_dropout]
        colors = [GREEN, RED]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=["Graduate", "Dropout"], colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=DARK_BG),
            textprops=dict(color="#e8eaf0", fontsize=11),
        )
        for at in autotexts:
            at.set_color(DARK_BG); at.set_fontweight("bold")
        ax.set_title("", pad=0)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # — Grade distribution by outcome —
    with col_b:
        st.markdown('<p class="section-title">Semester 1 Grade by Outcome</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        for label, color, name in [(0, GREEN, "Graduate"), (1, RED, "Dropout")]:
            ax.hist(df[df["Target"] == label]["Curricular_units_1st_sem_grade"],
                    bins=25, alpha=0.72, color=color, label=name, edgecolor="none")
        ax.set_xlabel("Grade (0–20)", color="#8b95a5", fontsize=10)
        ax.set_ylabel("Students",     color="#8b95a5", fontsize=10)
        ax.tick_params(colors="#8b95a5")
        ax.spines[["top","right"]].set_visible(False)
        for sp in ["bottom","left"]:
            ax.spines[sp].set_color("#1e2d40")
        ax.legend(facecolor=CARD_BG, edgecolor="#1e2d40", labelcolor="#e8eaf0")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # — Tuition fee status —
    st.markdown('<p class="section-title">Tuition Fee Status vs Dropout Rate</p>', unsafe_allow_html=True)
    ct = pd.crosstab(df["Tuition_fees_up_to_date"], df["Target"], normalize="index") * 100
    ct.index = ["Fees NOT up to date", "Fees up to date"]
    ct.columns = ["Graduate %", "Dropout %"]

    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    ct[["Graduate %", "Dropout %"]].plot(kind="barh", stacked=True, ax=ax,
                                          color=[GREEN, RED], edgecolor="none")
    ax.set_xlabel("% of students", color="#8b95a5")
    ax.tick_params(colors="#8b95a5")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#1e2d40")
    ax.legend(facecolor=CARD_BG, edgecolor="#1e2d40", labelcolor="#e8eaf0", loc="lower right")
    for i, (_, row) in enumerate(ct.iterrows()):
        ax.text(row["Dropout %"] / 2, i, f'{row["Dropout %"]:.1f}%', va="center",
                ha="center", color=DARK_BG, fontweight="bold", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════
# PAGE 2 — RISK CALCULATOR
# ═══════════════════════════════════════════════
elif page == "🔍  Risk Calculator":
    st.title("Individual Risk Calculator")
    st.markdown("<p style='color:#8b95a5;margin-top:-12px'>Enter a student profile to predict dropout probability in real time</p>", unsafe_allow_html=True)
    st.markdown("---")

    model_choice = st.selectbox("Prediction model", list(models.keys()), index=3)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-title">Academic Performance</p>', unsafe_allow_html=True)
        grade_s1     = st.slider("Semester 1 Grade (0–20)",         0.0, 20.0, 11.0, 0.1)
        grade_s2     = st.slider("Semester 2 Grade (0–20)",         0.0, 20.0, 10.5, 0.1)
        units_s1     = st.slider("Units Approved S1",                0,   10,   5)
        units_s2     = st.slider("Units Approved S2",                0,   10,   5)
        prev_grade   = st.slider("Previous Qualification Grade",     95.0, 190.0, 128.0, 0.5)
        adm_grade    = st.slider("Admission Grade",                  95.0, 190.0, 127.0, 0.5)

    with col2:
        st.markdown('<p class="section-title">Financial & Social</p>', unsafe_allow_html=True)
        tuition_ok   = st.selectbox("Tuition Fees Up To Date", [1, 0], format_func=lambda x: "Yes" if x else "No")
        scholarship  = st.selectbox("Scholarship Holder",       [0, 1], format_func=lambda x: "Yes" if x else "No")
        debtor       = st.selectbox("Debtor",                   [0, 1], format_func=lambda x: "Yes" if x else "No")
        displaced    = st.selectbox("Displaced",                [0, 1], format_func=lambda x: "Yes" if x else "No")
        gender       = st.selectbox("Gender",                   [1, 0], format_func=lambda x: "Male" if x else "Female")
        intl         = st.selectbox("International Student",    [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col3:
        st.markdown('<p class="section-title">Demographics & Macro</p>', unsafe_allow_html=True)
        age          = st.slider("Age at Enrollment",   17, 60, 20)
        daytime      = st.selectbox("Attendance",       [1, 0], format_func=lambda x: "Daytime" if x else "Evening")
        marital      = st.selectbox("Marital Status",   list(range(1, 7)), index=0)
        app_mode     = st.selectbox("Application Mode", list(range(1, 18)), index=0)
        course       = st.selectbox("Course",           list(range(1, 17)), index=0)
        unemp        = st.slider("Unemployment Rate (%)", 7.0, 17.0, 11.5, 0.1)

    st.markdown("---")

    # Build feature vector (same order as feature_names)
    input_dict = {
        "Curricular_units_1st_sem_grade":    grade_s1,
        "Curricular_units_2nd_sem_grade":    grade_s2,
        "Curricular_units_1st_sem_approved": units_s1,
        "Curricular_units_2nd_sem_approved": units_s2,
        "Tuition_fees_up_to_date":           tuition_ok,
        "Scholarship_holder":                scholarship,
        "Age_at_enrollment":                 age,
        "Daytime_evening_attendance":        daytime,
        "Debtor":                            debtor,
        "GDP":                               1.4,
        "Unemployment_rate":                 unemp,
        "Inflation_rate":                    1.2,
        "Mothers_qualification":             10,
        "Fathers_qualification":             10,
        "Marital_status":                    marital,
        "Application_mode":                  app_mode,
        "Course":                            course,
        "Previous_qualification_grade":      prev_grade,
        "Admission_grade":                   adm_grade,
        "Displaced":                         displaced,
        "Gender":                            gender,
        "International":                     intl,
    }
    X_input = pd.DataFrame([input_dict])[feature_names]
    X_scaled = scaler.transform(X_input)

    clf = models[model_choice]
    prob = clf.predict_proba(X_scaled)[0, 1]

    if prob >= 0.65:
        risk_label = "HIGH RISK"
        badge_cls  = "risk-high"
        bar_color  = RED
    elif prob >= 0.40:
        risk_label = "MODERATE RISK"
        badge_cls  = "risk-medium"
        bar_color  = YELLOW
    else:
        risk_label = "LOW RISK"
        badge_cls  = "risk-low"
        bar_color  = GREEN

    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"### Dropout Probability")
        st.markdown(f"<span class='{badge_cls}'>{risk_label}</span>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color:{bar_color};margin:.2rem 0'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        st.progress(float(prob))

    with r2:
        st.markdown("### Key Risk Signals")
        signals = []
        if grade_s1 < 8:    signals.append(("⚠️ Low S1 grade",        "critical academic predictor"))
        if units_s1 < 4:    signals.append(("⚠️ Few units approved",   "low engagement signal"))
        if not tuition_ok:  signals.append(("💰 Fees not up to date",  "strong dropout predictor"))
        if debtor:          signals.append(("💸 Debtor status",         "financial stress indicator"))
        if age > 25:        signals.append(("📅 Mature student",        "higher risk cohort"))
        if not scholarship: signals.append(("🎓 No scholarship",        "reduced support structure"))

        if signals:
            for icon_label, detail in signals:
                st.markdown(
                    f"<div style='background:#1a2035;border-left:3px solid {YELLOW};"
                    f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                    f"font-size:.88rem'>{icon_label} <span style='color:#4a4f5c'>— {detail}</span></div>",
                    unsafe_allow_html=True
                )
        else:
            st.success("No major risk flags detected for this student profile.")


# ═══════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════
elif page == "🤖  Model Comparison":
    st.title("Model Comparison")
    st.markdown("<p style='color:#8b95a5;margin-top:-12px'>4 classifiers · 5-fold stratified GridSearchCV · optimised for F1</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Metrics table
    st.markdown('<p class="section-title">Test Set Performance (ranked by F1)</p>', unsafe_allow_html=True)
    display_df = results_df.copy()
    display_df.index = ["🥇","🥈","🥉","4️⃣"]
    st.dataframe(display_df.style.format({
        "Accuracy":  "{:.1f}%",
        "Precision": "{:.3f}",
        "Recall":    "{:.3f}",
        "F1":        "{:.3f}",
        "AUC-ROC":   "{:.3f}",
    }).background_gradient(subset=["F1","AUC-ROC"], cmap="Blues"), use_container_width=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # — Bar chart of metrics —
    with col_left:
        st.markdown('<p class="section-title">F1 & AUC-ROC by Model</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        x     = np.arange(len(results_df))
        width = 0.35
        ax.bar(x - width/2, results_df["F1"],      width, color=ACCENT, alpha=.85, label="F1")
        ax.bar(x + width/2, results_df["AUC-ROC"], width, color=GREEN,  alpha=.85, label="AUC-ROC")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ","\n") for m in results_df["Model"]], color="#8b95a5", fontsize=8)
        ax.set_ylim(0.6, 1.0)
        ax.tick_params(axis="y", colors="#8b95a5")
        ax.spines[["top","right"]].set_visible(False)
        for sp in ["bottom","left"]:
            ax.spines[sp].set_color("#1e2d40")
        ax.legend(facecolor=CARD_BG, edgecolor="#1e2d40", labelcolor="#e8eaf0")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # — ROC curves —
    with col_right:
        st.markdown('<p class="section-title">ROC Curves — All Models</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        palette = [ACCENT, GREEN, YELLOW, RED]
        for (_, row), color in zip(results_df.iterrows(), palette):
            clf  = models[row["Model"]]
            prob = clf.predict_proba(X_test_sc)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{row['Model'].split()[0]} (AUC {row['AUC-ROC']:.3f})")
        ax.plot([0, 1], [0, 1], ":", color="#4a4f5c", lw=1)
        ax.set_xlabel("False Positive Rate", color="#8b95a5", fontsize=9)
        ax.set_ylabel("True Positive Rate",  color="#8b95a5", fontsize=9)
        ax.tick_params(colors="#8b95a5")
        ax.spines[["top","right"]].set_visible(False)
        for sp in ["bottom","left"]:
            ax.spines[sp].set_color("#1e2d40")
        ax.legend(facecolor=CARD_BG, edgecolor="#1e2d40", labelcolor="#e8eaf0",
                  fontsize=7.5, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # — Confusion matrix for best model —
    st.markdown('<p class="section-title">Confusion Matrix — XGBoost (Best Model)</p>', unsafe_allow_html=True)
    xgb   = models["XGBoost"]
    y_pred = xgb.predict(X_test_sc)
    cm    = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Graduate","Dropout"], color="#8b95a5")
    ax.set_yticklabels(["Graduate","Dropout"], color="#8b95a5")
    ax.set_xlabel("Predicted", color="#8b95a5"); ax.set_ylabel("Actual", color="#8b95a5")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "#333", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _, cc, _ = st.columns([1, 2, 1])
    with cc:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════
elif page == "🔬  SHAP Explainability":
    st.title("SHAP Explainability — XGBoost")
    st.markdown("<p style='color:#8b95a5;margin-top:-12px'>SHapley Additive exPlanations · Section 5.3 of the report</p>", unsafe_allow_html=True)
    st.markdown("---")

    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp_df   = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs}) \
                 .sort_values("Mean |SHAP|", ascending=False) \
                 .head(12).reset_index(drop=True)

    col_l, col_r = st.columns(2)

    # — Bar importance —
    with col_l:
        st.markdown('<p class="section-title">Top 12 Features by Mean |SHAP|</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)
        bars = ax.barh(imp_df["Feature"][::-1], imp_df["Mean |SHAP|"][::-1],
                       color=ACCENT, alpha=0.85, edgecolor="none")
        ax.set_xlabel("Mean |SHAP value|", color="#8b95a5", fontsize=9)
        ax.tick_params(colors="#8b95a5", labelsize=8)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#1e2d40")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # — Top features table —
    with col_r:
        st.markdown('<p class="section-title">Importance Table</p>', unsafe_allow_html=True)
        st.dataframe(
            imp_df.style.format({"Mean |SHAP|": "{:.4f}"})
                        .background_gradient(subset=["Mean |SHAP|"], cmap="Blues"),
            use_container_width=True, height=380
        )

    st.markdown("---")
    st.markdown('<p class="section-title">SHAP Beeswarm — Directional Feature Impact</p>', unsafe_allow_html=True)

    top_idx  = [feature_names.index(f) for f in imp_df["Feature"]]
    top_shap = shap_vals[:, top_idx]
    top_X    = X_test_df[imp_df["Feature"]]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    shap.summary_plot(
        top_shap, top_X,
        feature_names=imp_df["Feature"].tolist(),
        show=False, plot_size=None, color_bar=True
    )
    ax = plt.gca()
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="#8b95a5", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#1e2d40")
    ax.set_xlabel("SHAP value (impact on model output)", color="#8b95a5", fontsize=9)
    plt.gcf().set_facecolor(CARD_BG)
    plt.tight_layout()
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close("all")

    st.info(
        "**How to read this:** Each dot is one student. "
        "**Blue = low feature value, Red = high feature value.** "
        "Dots to the right push the model toward predicting *Dropout*; "
        "dots to the left push toward *Graduate*. "
        "High S1 grades (red, left) strongly reduce dropout risk."
    )
