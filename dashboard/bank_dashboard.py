import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Bank Deposit ML", page_icon="🏦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { padding: 1.5rem 2rem; }

.kpi {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #60a5fa;
    margin: 0;
}
.kpi-lbl { font-size: 0.75rem; color: #94a3b8; margin: 0; letter-spacing: 0.05em; }

.sec {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: #60a5fa;
    text-transform: uppercase;
    border-left: 3px solid #3266ad;
    padding-left: 0.6rem;
    margin: 1.4rem 0 0.6rem;
}

.note {
    background: #1e293b;
    border-left: 3px solid #f59e0b;
    padding: 0.5rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #fcd34d;
    margin-bottom: 1rem;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Bank Deposit ML")
    st.markdown("---")
    threshold = st.slider("Prediction threshold", 0.20, 0.80, 0.40, 0.05,
                          help="Lower = catch more subscribers, higher = more precise")
    st.markdown("---")
    selected_models = st.multiselect(
        "Models to show",
        ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
        default=["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    )
    st.markdown("---")
    st.caption("⚠️ `duration` excluded — only known after the call ends (data leakage).")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING  (duration excluded)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df_raw = pd.read_csv("bank.csv")
    return df_raw

@st.cache_data
def engineer(df_raw):
    df = df_raw.copy()

    # Outlier capping — duration removed
    for col in ["balance", "campaign", "previous"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

    # Log transforms — no duration_log
    df["balance_log"]  = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"].values))
    df["campaign_log"] = np.log1p(df["campaign"])

    # Age bins
    df["age_group"] = pd.cut(df["age"], bins=[0, 24, 35, 50, 60, 100],
                              labels=["student","early_career","mid_career",
                                      "pre_retirement","retired"])

    # Cyclical month/day
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    df["month_num"]      = df["month"].map(month_map)
    df["month_sin"]      = np.sin(2*np.pi*df["month_num"]/12)
    df["month_cos"]      = np.cos(2*np.pi*df["month_num"]/12)
    df["day_sin"]        = np.sin(2*np.pi*df["day"]/31)
    df["day_cos"]        = np.cos(2*np.pi*df["day"]/31)
    df["is_quarter_end"] = df["month_num"].isin([3,6,9,12]).astype(int)

    # Previous contact features
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["recency_score"]        = df.apply(
        lambda r: 1/(1+r["pdays"]) if r["pdays"] != -1 else 0, axis=1)
    df["poutcome_success"]     = (df["poutcome"] == "success").astype(int)

    # Interactions — no duration
    df["balance_x_housing"] = df["balance_log"] * (df["housing"] == "yes").astype(int)
    df["balance_per_age"]   = df["balance"] / (df["age"] + 1)

    # Target encoding
    for col in ["job","poutcome"]:
        deposit_target  = (df["deposit"] == "yes").astype(int)
        means           = deposit_target.groupby(df[col]).mean()
        df[col + "_te"] = df[col].map(means)
    df = df.drop(columns=["job","poutcome"])

    # Drop raw columns — duration dropped here
    df = df.drop(columns=["month","month_num","day","pdays",
                           "balance","campaign","duration"])

    # Ordinal encoding
    df["education"] = df["education"].map({"unknown":0,"primary":1,
                                            "secondary":2,"tertiary":3}).fillna(0).astype(int)
    df["age_group"]  = df["age_group"].astype(str).map(
        {"student":0,"early_career":1,"mid_career":2,
         "pre_retirement":3,"retired":4}).fillna(2).astype(int)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["marital","contact"], drop_first=True)

    # Binary columns
    for col in ["default","housing","loan"]:
        df[col] = (df[col] == "yes").astype(int)

    # Target
    df["deposit"] = (df["deposit"] == "yes").astype(int)
    return df

@st.cache_data
def train_models(df):
    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    configs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42,
                               C=1, penalty="l2", solver="liblinear",
                               class_weight="balanced"), True),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, max_depth=6,
                                   min_samples_leaf=10,
                                   class_weight="balanced"), False),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1,
                                   n_estimators=200, max_depth=10,
                                   min_samples_leaf=5,
                                   class_weight="balanced"), False),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="auc",
                          use_label_encoder=False,
                          scale_pos_weight=scale_pos_weight,
                          n_estimators=200, max_depth=5,
                          learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8), False),
    }

    results = {}
    for name, (m, scaled) in configs.items():
        Xtr = X_train_s if scaled else X_train
        Xte = X_test_s  if scaled else X_test
        m.fit(Xtr, y_train)
        y_prob = m.predict_proba(Xte)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = dict(
            model=m, y_prob=y_prob, fpr=fpr, tpr=tpr,
            auc=roc_auc_score(y_test, y_prob),
        )
    return results, X, X_test, y_test, scaler

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
try:
    df_raw = load_data()
except:
    st.error("Place `bank.csv` in the same folder as this script.")
    st.stop()

df       = engineer(df_raw)
results, X, X_test, y_test, scaler = train_models(df)
COLORS   = {"Logistic Regression":"#3b82f6","Decision Tree":"#f59e0b","Random Forest":"#10b981","XGBoost":"#a855f7"}
active   = {n: results[n] for n in selected_models if n in results}
best_name= max(results, key=lambda n: results[n]["auc"])

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#  Bank Deposit Subscription — ML Dashboard")


# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
k1,k2,k3,k4,k5,k6 = st.columns(6)
deposit_rate = (df_raw["deposit"]=="yes").mean()
best         = results[best_name]
y_pred_best  = (best["y_prob"] >= threshold).astype(int)

for col, val, lbl in zip(
    [k1,k2,k3,k4,k5,k6],
    [f"{len(df_raw):,}",
     f"{(df_raw['deposit']=='yes').sum():,}",
     f"{deposit_rate:.1%}",
     f"{X.shape[1]}",
     f"{best['auc']:.4f}",
     f"{accuracy_score(y_test, y_pred_best):.4f}"],
    ["TOTAL CLIENTS","SUBSCRIBED","DEPOSIT RATE",
     "FEATURES","BEST AUC","BEST ACCURACY"]
):
    with col:
        st.markdown(f"""<div class="kpi">
            <p class="kpi-val">{val}</p>
            <p class="kpi-lbl">{lbl}</p></div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — EDA  (left) | MODEL METRICS TABLE (right)
# ══════════════════════════════════════════════════════════════════════════════
col_eda, col_metrics = st.columns([1.1, 0.9])

with col_eda:
    st.markdown('<div class="sec">Exploratory Analysis</div>', unsafe_allow_html=True)
    eda_choice = st.selectbox("Variable", ["job","marital","education","contact","poutcome"],
                              label_visibility="collapsed")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#1e3a5f")

    # Univariate count
    counts = df_raw[eda_choice].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color="#3b82f6", edgecolor="none")
    axes[0].set_title(f"Count by {eda_choice}", color="#e2e8f0", fontsize=9)

    # Deposit rate
    rate = df_raw.groupby(eda_choice)["deposit"].apply(
        lambda x: (x=="yes").mean()).sort_values()
    colors_r = ["#10b981" if v >= deposit_rate else "#ef4444" for v in rate.values]
    axes[1].barh(rate.index, rate.values*100, color=colors_r, edgecolor="none")
    axes[1].axvline(deposit_rate*100, color="#f59e0b", linestyle="--", lw=1)
    axes[1].set_title("Deposit rate %", color="#e2e8f0", fontsize=9)

    plt.tight_layout(pad=1)
    st.pyplot(fig, use_container_width=True); plt.close()

with col_metrics:
    st.markdown('<div class="sec">Model Metrics (threshold={:.2f})</div>'.format(threshold),
                unsafe_allow_html=True)
    rows = []
    for name, r in active.items():
        yp = (r["y_prob"] >= threshold).astype(int)
        rows.append({
            "Model":     name,
            "AUC":       round(r["auc"], 4),
            "Accuracy":  round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, yp, zero_division=0), 4),
            "F1":        round(f1_score(y_test, yp, zero_division=0), 4),
        })
    mdf = pd.DataFrame(rows).set_index("Model")
    st.dataframe(mdf.style.highlight_max(axis=0, color="#1e3a5f")
                          .format("{:.4f}"),
                 use_container_width=True, height=160)

    # Metric bars
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")

    met_names = ["AUC","Accuracy","Precision","Recall","F1"]
    x = np.arange(len(met_names))
    w = 0.8 / max(len(active), 1)
    for i, (name, r) in enumerate(active.items()):
        yp = (r["y_prob"] >= threshold).astype(int)
        vals = [r["auc"],
                accuracy_score(y_test, yp),
                precision_score(y_test, yp, zero_division=0),
                recall_score(y_test, yp, zero_division=0),
                f1_score(y_test, yp, zero_division=0)]
        offset = (i - len(active)/2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=name,
               color=COLORS.get(name,"#60a5fa"), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(met_names, color="#e2e8f0", fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, labelcolor="#e2e8f0",
                                     facecolor="#1e293b", edgecolor="none")
    ax.set_title("Metrics comparison", color="#e2e8f0", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — ROC | CONFUSION MATRICES | THRESHOLD CURVE
# ══════════════════════════════════════════════════════════════════════════════
col_roc, col_cm, col_th = st.columns(3)

with col_roc:
    st.markdown('<div class="sec">ROC Curves</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.plot([0,1],[0,1],"--",color="#475569",lw=1)
    for name, r in active.items():
        ax.plot(r["fpr"], r["tpr"], color=COLORS.get(name,"#60a5fa"), lw=2,
                label=f"{name[:6]}.. {r['auc']:.3f}")
    ax.set_xlabel("FPR", color="#94a3b8", fontsize=8)
    ax.set_ylabel("TPR", color="#94a3b8", fontsize=8)
    ax.set_title("ROC Curves", color="#e2e8f0", fontsize=9)
    ax.legend(fontsize=7, labelcolor="#e2e8f0", facecolor="#1e293b", edgecolor="none")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

with col_cm:
    st.markdown('<div class="sec">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_model = st.selectbox("Model", list(active.keys()),
                             label_visibility="collapsed", key="cm_sel")
    if cm_model:
        yp = (active[cm_model]["y_prob"] >= threshold).astype(int)
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        cm = confusion_matrix(y_test, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No","Yes"], yticklabels=["No","Yes"],
                    linewidths=0.5)
        ax.set_title(f"{cm_model[:10]}.. t={threshold}", color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_xlabel("Predicted", color="#94a3b8", fontsize=8)
        ax.set_ylabel("Actual", color="#94a3b8", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

with col_th:
    st.markdown('<div class="sec">Threshold Curve</div>', unsafe_allow_html=True)
    th_model = st.selectbox("Model ", list(active.keys()),
                             label_visibility="collapsed", key="th_sel")
    if th_model:
        prob = active[th_model]["y_prob"]
        prec_a, rec_a, th_a = precision_recall_curve(y_test, prob)
        f1_a = 2*prec_a[:-1]*rec_a[:-1]/(prec_a[:-1]+rec_a[:-1]+1e-9)
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
        ax.plot(th_a, prec_a[:-1], color="#3b82f6", lw=2, label="Precision")
        ax.plot(th_a, rec_a[:-1],  color="#10b981", lw=2, label="Recall")
        ax.plot(th_a, f1_a,        color="#f59e0b", lw=2, label="F1")
        ax.axvline(threshold, color="#ef4444", linestyle="--", lw=1.5)
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.set_xlabel("Threshold", color="#94a3b8", fontsize=8)
        ax.set_title(f"P/R/F1 vs Threshold", color="#e2e8f0", fontsize=9)
        ax.legend(fontsize=7, labelcolor="#e2e8f0", facecolor="#1e293b", edgecolor="none")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — FEATURE IMPORTANCE | NUMERIC DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
col_fi, col_dist = st.columns([1.1, 0.9])

with col_fi:
    st.markdown('<div class="sec">Feature Importance</div>', unsafe_allow_html=True)
    fi_model = st.selectbox("Model  ", list(results.keys()),
                             label_visibility="collapsed", key="fi_sel")
    top_n = st.slider("Top N features", 5, X.shape[1], 12, key="fi_n")
    m = results[fi_model]["model"]

    if fi_model == "Logistic Regression":
        fi = pd.Series(m.coef_[0], index=X.columns)
        xlabel = "Coefficient"
    else:
        fi = pd.Series(m.feature_importances_, index=X.columns)
        xlabel = "Importance"

    fi_top = fi.reindex(fi.abs().sort_values(ascending=True).tail(top_n).index)
    colors_fi = ["#10b981" if v >= 0 else "#ef4444" for v in fi_top.values]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n*0.4)), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.barh(fi_top.index, fi_top.values, color=colors_fi, edgecolor="none")
    ax.axvline(0, color="#475569", lw=0.8)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=8)
    ax.set_title(f"{fi_model} — top {top_n} features", color="#e2e8f0", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

with col_dist:
    st.markdown('<div class="sec">Numeric Distribution</div>', unsafe_allow_html=True)
    num_col = st.selectbox("Variable ", ["age","balance","campaign","previous"],
                            label_visibility="collapsed", key="num_sel")
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")

    yes_data = df_raw[df_raw["deposit"]=="yes"][num_col]
    no_data  = df_raw[df_raw["deposit"]=="no"][num_col]

    axes[0].hist(no_data,  bins=25, alpha=0.6, color="#475569", label="No deposit")
    axes[0].hist(yes_data, bins=25, alpha=0.75, color="#3b82f6",label="Subscribed")
    axes[0].set_title(f"{num_col} distribution", color="#e2e8f0", fontsize=9)
    axes[0].legend(fontsize=7, labelcolor="#e2e8f0",
                   facecolor="#1e293b", edgecolor="none")

    bp = axes[1].boxplot([no_data, yes_data], labels=["No","Yes"],
                          patch_artist=True, widths=0.5,
                          medianprops=dict(color="#f59e0b", lw=2))
    bp["boxes"][0].set_facecolor("#475569"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#3b82f6"); bp["boxes"][1].set_alpha(0.7)
    for whisker in bp["whiskers"]: whisker.set_color("#94a3b8")
    for cap in bp["caps"]: cap.set_color("#94a3b8")
    axes[1].set_title(f"{num_col} boxplot", color="#e2e8f0", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — PREDICT NEW CLIENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">Predict for a New Client</div>', unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)
with p1:
    age      = st.number_input("Age", 18, 95, 40)
    job      = st.selectbox("Job", sorted(df_raw["job"].unique()))
    marital  = st.selectbox("Marital", df_raw["marital"].unique())
with p2:
    education= st.selectbox("Education", df_raw["education"].unique())
    balance  = st.number_input("Balance (€)", -5000, 50000, 1000)
    housing  = st.selectbox("Housing loan", ["yes","no"])
with p3:
    loan     = st.selectbox("Personal loan", ["yes","no"])
    default  = st.selectbox("Credit default", ["no","yes"])
    contact  = st.selectbox("Contact type", df_raw["contact"].unique())
with p4:
    campaign = st.number_input("Contacts this campaign", 1, 50, 2)
    pdays    = st.number_input("Days since last contact (-1=never)", -1, 999, -1)
    previous = st.number_input("Previous contacts", 0, 50, 0)
    poutcome = st.selectbox("Previous outcome", df_raw["poutcome"].unique())
    pred_model = st.selectbox("Model", list(results.keys()), key="pred_m")

if st.button("🔮  Predict", use_container_width=True):
    try:
        row = pd.DataFrame([{
            "age":age,"balance":balance,"campaign":campaign,
            "previous":previous,"job":job,"marital":marital,
            "education":education,"default":default,"housing":housing,
            "loan":loan,"contact":contact,"poutcome":poutcome,"pdays":pdays,
        }])

        # Feature engineering on single row
        for col in ["balance","campaign","previous"]:
            Q1,Q3 = df_raw[col].quantile(0.25), df_raw[col].quantile(0.75)
            IQR = Q3 - Q1
            row[col] = row[col].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)

        row["balance_log"]  = np.sign(row["balance"]) * np.log1p(np.abs(row["balance"].values))
        row["campaign_log"] = np.log1p(row["campaign"])
        row["age_group"]    = pd.cut(row["age"], bins=[0,24,35,50,60,100],
                                      labels=["student","early_career","mid_career",
                                              "pre_retirement","retired"])

        mn = 5  # default month = May
        row["month_sin"]      = np.sin(2*np.pi*mn/12)
        row["month_cos"]      = np.cos(2*np.pi*mn/12)
        row["day_sin"]        = np.sin(2*np.pi*15/31)
        row["day_cos"]        = np.cos(2*np.pi*15/31)
        row["is_quarter_end"] = 0

        row["was_contacted_before"] = int(pdays != -1)
        row["recency_score"]        = 1/(1+pdays) if pdays != -1 else 0
        row["poutcome_success"]     = int(poutcome == "success")

        row["balance_x_housing"] = row["balance_log"] * int(housing=="yes")
        row["balance_per_age"]   = row["balance"] / (age + 1)

        job_map  = df_raw.groupby("job")["deposit"].apply(lambda x: (x=="yes").mean())
        pout_map = df_raw.groupby("poutcome")["deposit"].apply(lambda x: (x=="yes").mean())
        row["job_te"]      = row["job"].map(job_map)
        row["poutcome_te"] = row["poutcome"].map(pout_map)

        row = row.drop(columns=["job","poutcome","balance","campaign","pdays"])

        row["education"] = row["education"].map({"unknown":0,"primary":1,
                                                  "secondary":2,"tertiary":3}).fillna(0).astype(int)
        row["age_group"]  = row["age_group"].astype(str).map(
            {"student":0,"early_career":1,"mid_career":2,
             "pre_retirement":3,"retired":4}).fillna(2).astype(int)

        for col in ["default","housing","loan"]:
            row[col] = (row[col]=="yes").astype(int)

        row = pd.get_dummies(row, columns=["marital","contact"])
        row = row.reindex(columns=X.columns, fill_value=0)

        m    = results[pred_model]["model"]
        Xin  = scaler.transform(row) if pred_model == "Logistic Regression" else row
        prob = m.predict_proba(Xin)[0][1]
        pred = int(prob >= threshold)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Subscription probability", f"{prob:.1%}")
        with r2:
            st.metric("Prediction (threshold)", "✅ Will subscribe" if pred else "❌ Won't subscribe")
        with r3:
            st.metric("Confidence", "High" if abs(prob-0.5) > 0.2 else "Moderate" if abs(prob-0.5) > 0.1 else "Low")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 1.2), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.barh(0, 1, color="#1e293b", height=0.5)
        ax.barh(0, prob, color="#10b981" if prob>=threshold else "#ef4444", height=0.5)
        ax.axvline(threshold, color="#f59e0b", lw=2, linestyle="--")
        ax.set_xlim(0,1); ax.set_yticks([])
        ax.set_xlabel("Probability", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
        ax.set_title(f"Predicted probability: {prob:.1%}  |  threshold: {threshold}",
                     color="#e2e8f0", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("Bank marketing dataset · LR / DT / RF / XGBoost models · duration excluded (data leakage prevention)")