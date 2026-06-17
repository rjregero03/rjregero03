"""
YTR Leadership Potential Prediction System — Upgraded Two-Layer Thesis Prototype
Thesis: Predicting Leadership Potential Among Youth on The Rock Members Using Machine Learning
School: Eulogio "Amang" Rodriguez Institute of Science and Technology (EARIST)
Degree: Bachelor of Science in Computer Science
Authors: Allera, Gabatino, Guiriba, Regero, Yadao (2027)

Main upgrade:
1) Stage 1 Behavior Model:
   leadership_behavior + godly_characteristics + patriotic_initiative -> behavior_label
2) Stage 2 Final Model:
   predicted behavior_label + mentor_evaluation + academic_excellence + attendance -> final_label
3) Rule-based safety layer after ML prediction.

Note: Synthetic data is for prototype validation only and must be replaced with actual YTR data.
"""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YTR Leadership Prediction System",
    page_icon="🌟",
    layout="wide",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

if is_dark:
    bg_main = "#0e1117"
    bg_secondary = "#161a24"
    bg_card = "#1a1d27"
    bg_input = "#212634"
    text_primary = "#f0f2f6"
    text_secondary = "#94a3b8"
    accent = "#4facfe"
    accent_light = "#00f2fe"
    border = "rgba(255,255,255,0.08)"
    shadow = "0 8px 32px rgba(0,0,0,0.30)"
    mpl_bg = "#1a1d27"
    mpl_text = "#f0f2f6"
    mpl_grid = "#2d3446"
    toggle_icon = "☀️"
    toggle_label = "Light Mode"
else:
    bg_main = "#f8fafc"
    bg_secondary = "#f1f5f9"
    bg_card = "#ffffff"
    bg_input = "#f8fafc"
    text_primary = "#0f172a"
    text_secondary = "#64748b"
    accent = "#4f46e5"
    accent_light = "#6366f1"
    border = "rgba(0,0,0,0.08)"
    shadow = "0 10px 40px rgba(0,0,0,0.06)"
    mpl_bg = "#ffffff"
    mpl_text = "#0f172a"
    mpl_grid = "#e2e8f0"
    toggle_icon = "🌙"
    toggle_label = "Dark Mode"

mpl.rcParams.update({
    "figure.facecolor": mpl_bg,
    "axes.facecolor": mpl_bg,
    "axes.edgecolor": mpl_grid,
    "axes.labelcolor": mpl_text,
    "axes.titlecolor": mpl_text,
    "xtick.color": mpl_text,
    "ytick.color": mpl_text,
    "text.color": mpl_text,
    "grid.color": mpl_grid,
    "legend.facecolor": mpl_bg,
    "legend.edgecolor": mpl_grid,
    "legend.labelcolor": mpl_text,
})

st.markdown("""
<style>
.stApp { font-family: sans-serif; }
[data-testid="stMetric"] { border-radius: 14px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
hdr_col, btn_col = st.columns([8, 1.4])
with hdr_col:
    st.title("🌟 YTR Leadership Potential Prediction System")
    st.markdown(
        "**Two-Layer Machine Learning Prototype for Youth on The Rock (YTR)**  \n"
        "*Stage 1: Behavioral Understanding → Stage 2: Final Leadership Decision → Rule Safety Layer*"
    )
with btn_col:
    st.markdown("<div style='padding-top:1.2rem'></div>", unsafe_allow_html=True)
    if st.button(f"{toggle_icon} {toggle_label}", width="stretch"):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

st.divider()

# ─────────────────────────────────────────────
# CONSTANTS AND HELPERS
# ─────────────────────────────────────────────
CLASS_NAMES_0 = ["Developing", "Moderate", "High"]
LABEL_MAP = {0: "🔴 Developing", 1: "🟡 Moderate", 2: "🟢 High"}
BEHAVIOR_MAP = {1: "🔴 Developing", 2: "🟡 Moderate", 3: "🟢 High"}

BEHAVIOR_FEATURES = ["leadership_behavior", "godly_characteristics", "patriotic_initiative"]
STAGE2_FEATURES = ["behavior_label_pred", "mentor_evaluation", "academic_excellence_norm", "attendance"]

# Exact thesis dataset columns requested for display and CSV export.
REQUESTED_DATASET_COLUMNS = [
    "leadership_behavior",
    "godly_characteristics",
    "patriotic_initiative",
    "behavior_score",
    "behavior_label",
    "mentor_evaluation",
    "academic_excellence",
    "attendance",
    "final_label",
]


def normalize_gpa(gpa):
    """Philippine GPA 1.0 best, 5.0 failed -> 1-5 normalized score where 5 is best."""
    return np.clip(6.0 - np.asarray(gpa, dtype=float), 1.0, 5.0)


def gpa_to_label(gpa: float) -> str:
    if gpa <= 1.00:
        return "Excellent"
    if gpa <= 1.50:
        return "Very Good"
    if gpa <= 2.00:
        return "Good"
    if gpa <= 2.50:
        return "Satisfactory"
    if gpa <= 3.00:
        return "Passing"
    if gpa < 5.00:
        return "Below Passing / Conditional"
    return "Failed"


def compute_behavior_score(leadership, godly, patriotic):
    return (0.50 * leadership) + (0.30 * godly) + (0.20 * patriotic)


def behavior_score_to_label(score):
    if score >= 3.75:
        return 3
    if score >= 2.55:
        return 2
    return 1


def decode_label_zero_based(label):
    return LABEL_MAP[int(label)]


def decode_behavior_label(label):
    return BEHAVIOR_MAP[int(label)]


def apply_rules_v2(attendance, gpa, mentor, behavior_score, prediction):
    """
    Rule-based safety layer after Stage 2 ML prediction.
    Rules are intentionally limited to extreme/logical edge cases so they do not dominate ML.
    prediction uses 0=Developing, 1=Moderate, 2=High.
    """
    applied = []
    final_prediction = int(prediction)

    if gpa >= 5.0:
        return 0, ["Failed GPA (5.0) → automatically classified as Developing."]

    if attendance < 20:
        return 0, ["Very low attendance (<20%) → automatically classified as Developing."]

    if attendance < 40 and final_prediction == 2:
        final_prediction = 1
        applied.append("Attendance below 40% → cannot be classified as High; adjusted to Moderate.")

    if behavior_score < 2.0 and final_prediction == 2:
        final_prediction = 1
        applied.append("Behavior score below 2.0 → cannot be classified as High; adjusted to Moderate.")

    if mentor <= 2.0 and final_prediction == 2:
        final_prediction = 1
        applied.append("Mentor evaluation ≤ 2.0 → cannot be classified as High; adjusted to Moderate.")

    if mentor >= 4.5 and final_prediction == 0 and attendance >= 40 and gpa < 5.0:
        final_prediction = 1
        applied.append("Strong mentor evaluation ≥ 4.5 → allowed upgrade from Developing to Moderate.")

    return final_prediction, applied

# ─────────────────────────────────────────────
# REALISTIC SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_realistic_dataset(n=900, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # Irregular cluster mixture, not perfect class profiles.
    cluster_probs = [0.30, 0.45, 0.25]
    latent_classes = rng.choice([0, 1, 2], size=n, p=cluster_probs)

    for cls in latent_classes:
        # Shared latent character with overlap and skew.
        if cls == 0:      # Developing-leaning
            base = rng.beta(2.0, 4.0) * 4 + 1
        elif cls == 1:    # Moderate-leaning
            base = rng.beta(3.2, 3.0) * 4 + 1
        else:             # High-leaning
            base = rng.beta(4.2, 2.2) * 4 + 1

        leadership = np.clip(base + rng.normal(0, 0.55), 1, 5)
        # Slight correlation leadership ↔ godly, not perfect.
        godly = np.clip(0.45 * leadership + 0.55 * (rng.beta(3, 2.6) * 4 + 1) + rng.normal(0, 0.45), 1, 5)
        patriotic = np.clip(0.25 * leadership + 0.75 * (rng.beta(2.7, 2.8) * 4 + 1) + rng.normal(0, 0.60), 1, 5)

        behavior_score = compute_behavior_score(leadership, godly, patriotic)
        behavior_label = behavior_score_to_label(behavior_score)

        # Mentor is partially independent and can contradict behavior.
        mentor = np.clip(
            0.55 * behavior_score + 0.45 * (rng.beta(3.0, 2.8) * 4 + 1) + rng.normal(0, 0.70),
            1,
            5,
        )

        # Academic GPA: lower is better. Weakly related only.
        gpa_center = 3.35 - (0.26 * behavior_score) + rng.normal(0, 0.38)
        gpa = np.clip(gpa_center, 1, 5)

        # Attendance: skewed and overlapping.
        attendance = np.clip(42 + (behavior_score * 9.5) + rng.normal(0, 16), 0, 100)

        # Inject realistic contradictions and borderline cases.
        event = rng.random()
        if event < 0.08:
            # High GPA but low leadership / low engagement.
            gpa = np.clip(rng.normal(1.55, 0.25), 1, 2.2)
            leadership = np.clip(rng.normal(2.1, 0.45), 1, 3.0)
        elif event < 0.16:
            # Low GPA but high mentor evaluation.
            gpa = np.clip(rng.normal(3.2, 0.40), 2.6, 4.5)
            mentor = np.clip(rng.normal(4.35, 0.35), 3.4, 5)
        elif event < 0.24:
            # High attendance but low engagement.
            attendance = np.clip(rng.normal(90, 5), 75, 100)
            leadership = np.clip(rng.normal(2.35, 0.45), 1, 3.2)
            patriotic = np.clip(rng.normal(2.4, 0.55), 1, 3.4)
        elif event < 0.32:
            # Good behavior but low attendance.
            attendance = np.clip(rng.normal(36, 8), 5, 55)
            leadership = np.clip(rng.normal(4.0, 0.45), 2.8, 5)
            godly = np.clip(rng.normal(4.1, 0.45), 2.8, 5)

        behavior_score = compute_behavior_score(leadership, godly, patriotic)
        behavior_label = behavior_score_to_label(behavior_score)
        academic_excellence_norm = float(normalize_gpa(gpa))

        # Final score emphasizes behavior + mentor, reduces GPA/attendance dominance.
        final_score = (
            0.42 * behavior_label +
            0.32 * mentor +
            0.14 * academic_excellence_norm +
            0.12 * (attendance / 20.0) +
            rng.normal(0, 0.42)
        )

        if final_score >= 3.95:
            final_label = 2
        elif final_score >= 2.85:
            final_label = 1
        else:
            final_label = 0

        # Ground-truth logical corrections, but not too dominant.
        if gpa >= 5.0 or attendance < 20:
            final_label = 0
        if mentor <= 1.8 and final_label == 2:
            final_label = 1
        if attendance < 40 and final_label == 2:
            final_label = 1

        rows.append({
            "leadership_behavior": round(float(leadership), 2),
            "godly_characteristics": round(float(godly), 2),
            "patriotic_initiative": round(float(patriotic), 2),
            "mentor_evaluation": round(float(mentor), 2),
            "academic_excellence": round(float(gpa), 2),
            "academic_excellence_norm": round(float(academic_excellence_norm), 3),
            "attendance": round(float(attendance), 1),
            "behavior_score": round(float(behavior_score), 3),
            "behavior_label": int(behavior_label),
            "final_label": int(final_label),
        })

    df = pd.DataFrame(rows)

    # Add a small amount of label noise to reduce artificial perfection.
    noise_idx = df.sample(frac=0.035, random_state=seed + 9).index
    for idx in noise_idx:
        current = df.at[idx, "final_label"]
        if current == 0:
            df.at[idx, "final_label"] = 1
        elif current == 2:
            df.at[idx, "final_label"] = 1
        else:
            df.at[idx, "final_label"] = int(rng.choice([0, 2]))

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


df = generate_realistic_dataset()

# Thesis-ready dataset export with the exact requested column order.
export_df = df[REQUESTED_DATASET_COLUMNS].copy()
export_df.to_csv("ytr_synthetic_dataset.csv", index=False)

# ─────────────────────────────────────────────
# STUDY OVERVIEW
# ─────────────────────────────────────────────
with st.expander("📌 Study Overview — Two-Layer ML Design", expanded=False):
    st.markdown("""
    ### Why this version is stronger
    This upgraded system separates the prediction into two academic layers:

    1. **Stage 1 — Behavior Model**  
       Uses only the three behavioral traits: leadership behavior, godly characteristics, and patriotic initiative.  
       Output: **behavior_label** and **behavior_score**.

    2. **Stage 2 — Final Leadership Model**  
       Uses the predicted behavior result together with mentor evaluation, GPA, and attendance.  
       Output: **final_label**.

    3. **Rule-Based Safety Layer**  
       Corrects only extreme cases, such as failed GPA, very low attendance, very low mentor evaluation, or logical violations.

    This structure is more defendable because it mimics real YTR evaluation: **behavior is evaluated first**, then leadership readiness is decided using mentor and organizational records.
    """)

# ─────────────────────────────────────────────
# DATASET PREVIEW
# ─────────────────────────────────────────────
st.subheader("📊 Dataset Overview — Realistic Synthetic Prototype Data")
st.caption("The dataset includes overlap, contradictions, borderline samples, and small label noise. Replace this with actual YTR survey data for deployment.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Samples", len(df))
with c2:
    st.metric("Input Features", "6")
with c3:
    st.metric("Pipeline", "2 Layers")
with c4:
    st.metric("Validation", "10-Fold CV")

preview = df[REQUESTED_DATASET_COLUMNS].copy()
preview["behavior_label"] = [BEHAVIOR_MAP[x] for x in preview["behavior_label"]]
preview["final_label"] = [LABEL_MAP[x] for x in preview["final_label"]]
st.dataframe(preview.head(20), width="stretch", hide_index=True)

st.download_button(
    "⬇️ Download Thesis Dataset CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="ytr_synthetic_dataset.csv",
    mime="text/csv",
    width="stretch",
)

left, right = st.columns([1.2, 1])
with left:
    class_counts = df["final_label"].value_counts().sort_index()
    dist_df = pd.DataFrame({
        "Class": [LABEL_MAP[i] for i in [0, 1, 2]],
        "Count": [int(class_counts.get(i, 0) or 0) for i in [0, 1, 2]],
        "Percentage": [f"{(class_counts.get(i, 0) or 0) / len(df) * 100:.1f}%" for i in [0, 1, 2]],
    })
    st.markdown("**Final Label Distribution**")
    st.dataframe(dist_df, width="stretch", hide_index=True)
with right:
    st.info(
        "**Dataset note:** This is not final real-world accuracy. It is prototype validation using synthetic data designed to be less clean and more realistic."
    )

# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────
st.divider()
st.subheader("⚙️ Two-Layer Training Pipeline")

# Split once and preserve indices for proper staged training.
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=df["final_label"],
)
train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

X1_train = train_df[BEHAVIOR_FEATURES]
y1_train = train_df["behavior_label"]
X1_test = test_df[BEHAVIOR_FEATURES]
y1_test = test_df["behavior_label"]

stage1_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial"),
    "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=8, min_samples_split=8, class_weight="balanced", random_state=42),
}

stage1_results = []
for name, model in stage1_models.items():
    model.fit(X1_train, y1_train)
    pred = model.predict(X1_test)
    stage1_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y1_test, pred),
        "Precision": precision_score(y1_test, pred, average="weighted", zero_division=0.0),
        "Recall": recall_score(y1_test, pred, average="weighted", zero_division=0.0),
        "F1-score": f1_score(y1_test, pred, average="weighted", zero_division=0.0),
        "_model": model,
        "_pred": pred,
    })

stage1_metrics = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in stage1_results]).set_index("Model")
best_stage1_name = stage1_metrics["F1-score"].idxmax()
best_stage1_model = [r["_model"] for r in stage1_results if r["Model"] == best_stage1_name][0]

# Predicted behavior labels become the input for Stage 2.
train_df["behavior_label_pred"] = best_stage1_model.predict(train_df[BEHAVIOR_FEATURES])
test_df["behavior_label_pred"] = best_stage1_model.predict(test_df[BEHAVIOR_FEATURES])

X2_train = train_df[STAGE2_FEATURES]
y2_train = train_df["final_label"]
X2_test = test_df[STAGE2_FEATURES]
y2_test = test_df["final_label"]

stage2_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial"),
    "Random Forest": RandomForestClassifier(n_estimators=260, max_depth=9, min_samples_split=7, min_samples_leaf=2, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.90,
        reg_alpha=0.2,
        reg_lambda=1.4,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    ),
}

stage2_results = []
for name, model in stage2_models.items():
    model.fit(X2_train, y2_train)
    pred = model.predict(X2_test)
    stage2_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y2_test, pred),
        "Precision": precision_score(y2_test, pred, average="weighted", zero_division=0.0),
        "Recall": recall_score(y2_test, pred, average="weighted", zero_division=0.0),
        "F1-score": f1_score(y2_test, pred, average="weighted", zero_division=0.0),
        "_model": model,
        "_pred": pred,
    })

stage2_metrics = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in stage2_results]).set_index("Model")
best_stage2_name = stage2_metrics["F1-score"].idxmax()
best_stage2_model = [r["_model"] for r in stage2_results if r["Model"] == best_stage2_name][0]
best_stage2_pred = [r["_pred"] for r in stage2_results if r["Model"] == best_stage2_name][0]

# Rule-adjusted pipeline accuracy.
rule_adjusted_preds = []
for _, row in test_df.iterrows():
    raw_pred = best_stage2_model.predict(pd.DataFrame([row[STAGE2_FEATURES]]))[0]
    adjusted, _ = apply_rules_v2(
        attendance=row["attendance"],
        gpa=row["academic_excellence"],
        mentor=row["mentor_evaluation"],
        behavior_score=row["behavior_score"],
        prediction=raw_pred,
    )
    rule_adjusted_preds.append(adjusted)

pipeline_accuracy = accuracy_score(y2_test, rule_adjusted_preds)
pipeline_f1 = f1_score(y2_test, rule_adjusted_preds, average="weighted", zero_division=0.0)

results_summary = pd.DataFrame([
    {"Section": "Stage 1 Behavior Model", "Best Model": best_stage1_name, "Accuracy": stage1_metrics.loc[best_stage1_name, "Accuracy"], "F1-score": stage1_metrics.loc[best_stage1_name, "F1-score"]},
    {"Section": "Stage 2 Final Model", "Best Model": best_stage2_name, "Accuracy": stage2_metrics.loc[best_stage2_name, "Accuracy"], "F1-score": stage2_metrics.loc[best_stage2_name, "F1-score"]},
    {"Section": "Full Rule-Adjusted Pipeline", "Best Model": f"{best_stage1_name} + {best_stage2_name}", "Accuracy": pipeline_accuracy, "F1-score": pipeline_f1},
])
results_summary.to_csv("ytr_model_results_summary.csv", index=False)

# K-fold functions.
def run_stage1_cv(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    return cross_val_score(clone(model), X, y, cv=skf, scoring="accuracy").mean()


def run_full_pipeline_cv(stage1_model, stage2_model, full_df):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accs = []
    f1s = []
    y_all = full_df["final_label"].values

    for tr_idx, te_idx in skf.split(full_df, y_all):
        tr = full_df.iloc[tr_idx].copy()
        te = full_df.iloc[te_idx].copy()

        s1 = clone(stage1_model)
        s1.fit(tr[BEHAVIOR_FEATURES], tr["behavior_label"])

        tr["behavior_label_pred"] = s1.predict(tr[BEHAVIOR_FEATURES])
        te["behavior_label_pred"] = s1.predict(te[BEHAVIOR_FEATURES])

        s2 = clone(stage2_model)
        s2.fit(tr[STAGE2_FEATURES], tr["final_label"])
        preds = s2.predict(te[STAGE2_FEATURES])

        adjusted = []
        for pred, (_, row) in zip(preds, te.iterrows()):
            new_pred, _ = apply_rules_v2(
                row["attendance"], row["academic_excellence"], row["mentor_evaluation"], row["behavior_score"], pred
            )
            adjusted.append(new_pred)

        accs.append(accuracy_score(te["final_label"], adjusted))
        f1s.append(f1_score(te["final_label"], adjusted, average="weighted", zero_division=0.0))

    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))

with st.spinner("Running 10-Fold Cross-Validation for the two-layer pipeline..."):
    stage1_cv_acc = run_stage1_cv(best_stage1_model, df[BEHAVIOR_FEATURES], df["behavior_label"])
    pipeline_cv_acc, pipeline_cv_std, pipeline_cv_f1 = run_full_pipeline_cv(best_stage1_model, best_stage2_model, df)

# Display evaluation.
tab1, tab2, tab3 = st.tabs(["🧠 Stage 1 Behavior Model", "🎯 Stage 2 Final Model", "🔁 Overall Pipeline CV"])

with tab1:
    st.markdown("**Stage 1 Input:** leadership_behavior + godly_characteristics + patriotic_initiative")
    st.dataframe(stage1_metrics.style.format("{:.4f}").highlight_max(axis=0), width="stretch")  # type: ignore
    st.success(f"Best Stage 1 Model: **{best_stage1_name}** | 10-Fold CV Accuracy: **{stage1_cv_acc:.4f}**")

with tab2:
    st.markdown("**Stage 2 Input:** predicted behavior_label + mentor_evaluation + academic_excellence + attendance")
    st.dataframe(stage2_metrics.style.format("{:.4f}").highlight_max(axis=0), width="stretch")  # type: ignore
    st.success(f"Best Stage 2 Model: **{best_stage2_name}** | Test Accuracy before rules: **{accuracy_score(y2_test, best_stage2_pred):.4f}**")

with tab3:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Pipeline Test Accuracy", f"{pipeline_accuracy:.2%}")
    with m2:
        st.metric("Pipeline Test F1", f"{pipeline_f1:.2%}")
    with m3:
        st.metric("10-Fold CV Accuracy", f"{pipeline_cv_acc:.2%}", f"± {pipeline_cv_std:.2%}")
    st.info(
        "The reported accuracy is based on synthetic prototype data. It is useful for system validation, but it should not be claimed as final real-world accuracy until tested with actual YTR records."
    )
    st.download_button(
        "⬇️ Download Model Results Summary CSV",
        data=results_summary.to_csv(index=False).encode("utf-8"),
        file_name="ytr_model_results_summary.csv",
        mime="text/csv",
        width="stretch",
    )

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
st.divider()
st.subheader("📈 Visual Analytics")

v1, v2 = st.columns(2)
with v1:
    st.markdown("**Stage 2 Model Comparison**")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(stage2_metrics.index))
    width = 0.2
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score"]
    for i, col in enumerate(metric_cols):
        ax.bar(x + (i - 1.5) * width, stage2_metrics[col], width, label=col, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stage2_metrics.index, rotation=12, ha="right")
    ax.set_ylim(0.60, 1.05)
    ax.axhline(0.85, linestyle="--", linewidth=1.2, alpha=0.75)
    ax.set_ylabel("Score")
    ax.set_title("Final Model Metrics")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

with v2:
    st.markdown(f"**Confusion Matrix — Final Pipeline ({best_stage2_name})**")
    cm = confusion_matrix(y2_test, rule_adjusted_preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES_0)
    ax.set_yticklabels(CLASS_NAMES_0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    threshold = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > threshold else mpl_text, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

r1, r2 = st.columns(2)
with r1:
    st.markdown("**Per-Class Classification Report — Rule Adjusted Pipeline**")
    report = classification_report(y2_test, rule_adjusted_preds, target_names=CLASS_NAMES_0, output_dict=True)
    report_df = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].astype(int)
    st.dataframe(report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), width="stretch")

with r2:
    st.markdown("**Feature Importance — Stage 2 Random Forest / XGBoost**")
    importance_model = best_stage2_model
    if hasattr(importance_model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": ["Behavior Label", "Mentor Evaluation", "Academic Excellence", "Attendance"],
            "Importance": importance_model.feature_importances_,
        }).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(imp["Feature"], imp["Importance"], alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(f"{best_stage2_name} Feature Importance")
        ax.grid(axis="x", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
    else:
        coef = np.mean(np.abs(best_stage2_model.coef_), axis=0)
        coef_df = pd.DataFrame({
            "Feature": ["Behavior Label", "Mentor Evaluation", "Academic Excellence", "Attendance"],
            "Coefficient Strength": coef,
        }).sort_values("Coefficient Strength")
        st.dataframe(coef_df, width="stretch", hide_index=True)

# ─────────────────────────────────────────────
# PREDICTION UI
# ─────────────────────────────────────────────
st.divider()
st.subheader("🔍 Predict a YTR Member's Leadership Potential")
st.caption("Stage 1 predicts behavior first. Stage 2 uses that behavior output with mentor, GPA, and attendance to produce the final leadership prediction.")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    st.markdown("**🧭 Behavioral Attributes**")
    inp_lead = st.slider("Leadership Behavior (1–5)", 1.0, 5.0, 3.4, step=0.1)
    inp_god = st.slider("Godly Characteristics (1–5)", 1.0, 5.0, 3.5, step=0.1)
    inp_pat = st.slider("Patriotic Initiative (1–5)", 1.0, 5.0, 3.3, step=0.1)
with pc2:
    st.markdown("**👨‍🏫 Mentor + Attendance**")
    inp_mentor = st.slider("Mentor Evaluation (1–5)", 1.0, 5.0, 3.6, step=0.1)
    inp_att = st.slider("Attendance / Participation (%)", 0, 100, 78, step=1)
with pc3:
    st.markdown("**📚 Academic Excellence**")
    inp_gpa = st.select_slider(
        "GPA (1.00 = Excellent → 5.00 = Failed)",
        options=[round(x * 0.25, 2) for x in range(4, 21)],
        value=2.25,
    )
    st.markdown(f"<div class='card'><b>GPA Meaning:</b><br>{inp_gpa:.2f} — {gpa_to_label(inp_gpa)}</div>", unsafe_allow_html=True)

if st.button("🔮 Predict Leadership Potential", type="primary", width="stretch"):
    user_behavior_score = compute_behavior_score(inp_lead, inp_god, inp_pat)
    user_behavior_rule_label = behavior_score_to_label(user_behavior_score)

    stage1_input = pd.DataFrame([{
        "leadership_behavior": inp_lead,
        "godly_characteristics": inp_god,
        "patriotic_initiative": inp_pat,
    }])
    predicted_behavior_label = int(best_stage1_model.predict(stage1_input)[0])

    stage2_input = pd.DataFrame([{
        "behavior_label_pred": predicted_behavior_label,
        "mentor_evaluation": inp_mentor,
        "academic_excellence_norm": float(normalize_gpa(inp_gpa)),
        "attendance": inp_att,
    }])

    model_predictions = {}
    model_confidences = {}
    for name, model in stage2_models.items():
        pred = int(model.predict(stage2_input)[0])
        model_predictions[name] = pred
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(stage2_input)[0]
            model_confidences[name] = float(np.max(proba))
        else:
            model_confidences[name] = np.nan

    raw_final_prediction = int(best_stage2_model.predict(stage2_input)[0])
    final_prediction, rule_messages = apply_rules_v2(
        attendance=inp_att,
        gpa=inp_gpa,
        mentor=inp_mentor,
        behavior_score=user_behavior_score,
        prediction=raw_final_prediction,
    )

    st.subheader("📋 Prediction Results")

    a, b, c, d = st.columns(4)
    with a:
        st.metric("Stage 1 Behavior Score", f"{user_behavior_score:.2f}/5")
    with b:
        st.metric("Stage 1 Behavior Label", decode_behavior_label(predicted_behavior_label))
    with c:
        st.metric("Stage 2 ML Decision", decode_label_zero_based(raw_final_prediction))
    with d:
        st.metric("Final Decision", decode_label_zero_based(final_prediction))

    st.markdown("**Model Votes — Stage 2**")
    mv1, mv2, mv3 = st.columns(3)
    for col, (name, pred) in zip([mv1, mv2, mv3], model_predictions.items()):
        conf = model_confidences[name]
        delta = f"Confidence: {conf:.1%}" if not np.isnan(conf) else ""
        with col:
            st.metric(name, decode_label_zero_based(pred), delta)

    if rule_messages:
        for msg in rule_messages:
            st.warning(f"🛡️ Rule Applied: {msg}")
    else:
        st.success("✅ No rule override applied. Final result follows the two-layer ML pipeline.")

    if final_prediction == 0:
        st.info("🔴 **Developing** — The member may need more mentoring, consistency, and guided leadership exposure before being recommended for major roles.")
    elif final_prediction == 1:
        st.info("🟡 **Moderate** — The member shows potential and may benefit from targeted responsibilities, coaching, and leadership practice.")
    else:
        st.info("🟢 **High** — The member shows strong leadership readiness and may be considered for leadership responsibilities, while still requiring human validation.")

    with st.expander("🧠 Explain this prediction"):
        st.markdown(f"""
        - **Behavior score** was computed using: Leadership 50%, Godly Characteristics 30%, Patriotic Initiative 20%.
        - **Stage 1** predicted the behavior level as **{decode_behavior_label(predicted_behavior_label)}**.
        - **Stage 2** used behavior result + mentor evaluation + GPA + attendance.
        - GPA **{inp_gpa:.2f}** was normalized to **{float(normalize_gpa(inp_gpa)):.2f}/5** because Philippine GPA uses 1.0 as highest and 5.0 as failed.
        - The final safety layer only changes the result when there are extreme cases or logical violations.
        """)

# ─────────────────────────────────────────────
# DEFENSE GUIDE
# ─────────────────────────────────────────────
st.divider()
with st.expander("🎓 Defense Explanation Script", expanded=False):
    st.markdown("""
    **How to explain the system:**

    “Our system uses a two-layer machine learning pipeline. The first layer predicts the member’s behavioral readiness using leadership behavior, godly characteristics, and patriotic initiative. After that, the second layer uses the predicted behavior result together with mentor evaluation, academic excellence, and attendance to produce the final leadership potential label.”

    **Why this is better:**

    “This design prevents the model from relying too much on GPA or attendance. It follows a more realistic YTR evaluation process because behavior is assessed first, then the final decision considers mentor and organizational records.”

    **Why synthetic data is acceptable for prototype:**

    “Since we do not yet have complete real-world YTR records, we generated synthetic data with overlapping values, contradictions, borderline members, and small label noise. This makes the prototype less artificial, but the final deployment should still be validated using actual YTR data.”

    **Role of rules:**

    “The rules are not the main decision-maker. They only correct extreme cases, such as failed GPA, very low attendance, or mentor evaluation that contradicts a high prediction.”
    """)

st.caption(
    "YTR Leadership Potential Prediction System | Two-Layer ML Prototype | RF · XGBoost · Logistic Regression | Synthetic data for prototype validation only"
)
