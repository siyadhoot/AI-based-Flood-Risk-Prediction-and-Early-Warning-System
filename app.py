from pathlib import Path
import json
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.ann.inference import predict_ann

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "processed" / "flood_model_data.csv",
    BASE_DIR / "data" / "processed" / "flood_cleaned.csv",
    BASE_DIR / "data" / "processed" / "flood_combined.csv",
    BASE_DIR / "data" / "combined_flood_dataset.csv",
]
METADATA_PATH = BASE_DIR / "models" / "training_metadata.json"
RESULTS_CANDIDATES = [
    BASE_DIR / "outputs" / "model_evaluation" / "tuned" / "model_comparison.csv",
    BASE_DIR / "outputs" / "ann" / "model_comparison.csv",
    BASE_DIR / "outputs" / "model_evaluation" / "model_comparison.csv",
    BASE_DIR / "models" / "model_comparison.csv",
]
EVAL_DIR = BASE_DIR / "outputs" / "model_evaluation"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="AI-Based Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp, [data-testid="stAppViewContainer"] {background: #f4f7fb; color: #18283a;}
      [data-testid="stHeader"] {background: transparent;}
      .block-container {max-width: 1500px; padding: 1.45rem 2.2rem 2.4rem;}
      [data-testid="stSidebar"] {background: #101f33; border-right: 1px solid #20344d;}
      [data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding: 1.4rem 1.05rem;}
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label {color: #e9eff6;}
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #aebbd0;}
      [data-testid="stSidebar"] hr {border-color: #30445c;}
      .hero {display: flex; align-items: center; gap: .85rem; padding: 1.05rem 1.25rem;
        border-radius: 12px; background: #fff; border: 1px solid #dce4ee;
        box-shadow: 0 2px 8px rgba(18, 43, 70, .045); margin-bottom: .8rem;}
      .hero-icon {display: flex; align-items: center; color: #197c91;}
      .hero h1 {font-size: clamp(1.45rem, 2.3vw, 2rem); letter-spacing: -.025em;
        margin: 0 0 .23rem 0; color: #14263b;}
      .hero-subtitle {color: #63748a; font-size: .92rem;}
      .prototype-note {padding: .75rem 1rem; background: #eaf5f5; border: 1px solid #d0e7e5;
        border-left: 4px solid #238a8d; border-radius: 8px; color: #24474b;
        font-size: .91rem; line-height: 1.5; margin: 0 0 1.4rem;}
      [data-testid="stMetric"] {padding: .85rem 1rem; background: #fff; border: 1px solid #dce4ee;
        border-radius: 10px; box-shadow: 0 2px 7px rgba(18, 43, 70, .035);}
      [data-testid="stMetricLabel"] p {color: #5b6e83; font-size: .86rem;}
      [data-testid="stMetricValue"] {color: #183c5a; font-weight: 700;}
      [data-testid="stTabs"] button[role="tab"],
      [data-testid="stTabs"] button[role="tab"] * {color: #173b5e !important; opacity: 1 !important;}
      [data-baseweb="tab-list"] [data-baseweb="tab"],
      [data-baseweb="tab-list"] [data-baseweb="tab"] *,
      button[data-baseweb="tab"], button[data-baseweb="tab"] *,
      [role="tab"], [role="tab"] * {
        color: #173b5e !important;
        -webkit-text-fill-color: #173b5e !important;
        opacity: 1 !important;
      }
      [data-testid="stTabs"] button[role="tab"] {padding: .7rem 1.1rem;
        font-weight: 600; border-bottom: 2px solid transparent;}
      [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color: #075c83 !important;
        border-bottom-color: #1a8998;}
      [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"],
      [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] *,
      button[data-baseweb="tab"][aria-selected="true"],
      button[data-baseweb="tab"][aria-selected="true"] *,
      [role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] * {
        color: #075c83 !important;
        -webkit-text-fill-color: #075c83 !important;
      }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] {background: #168394;}
      [data-testid="stDataFrame"] {border: 1px solid #dce4ee; border-radius: 8px; overflow: hidden;}
      [data-testid="stImage"] img {border: 1px solid #dce4ee; border-radius: 8px;}
      .status-line {display: flex; align-items: flex-start; gap: .55rem; margin: .62rem 0;
        color: #e6edf6; font-size: .91rem; line-height: 1.35;}
      .status-check {color: #55d69a; font-weight: 700; min-width: 1rem;}
      .status-missing {color: #f18b89; font-weight: 700; min-width: 1rem;}
      .status-detail {color: #aebbd0; font-size: .76rem; margin: -.3rem 0 .58rem 1.55rem;
        overflow-wrap: anywhere;}
      .side-target {padding: .85rem .9rem; background: #172a41; border: 1px solid #2b4059;
        border-radius: 9px; margin-top: .65rem;}
      .side-target-label {color: #9eb0c5; font-size: .72rem; text-transform: uppercase;
        letter-spacing: .08em; margin-bottom: .3rem;}
      .side-target-value {color: #f2f6fa; font-weight: 650;}
      .side-target-caption {color: #b6c5d6; font-size: .8rem; margin-top: .25rem;}
      .prediction {padding: .95rem 1rem; border-radius: 9px; background: #edf6f7;
        border: 1px solid #cfe4e5; border-left: 4px solid #278b91; margin: .5rem 0 1rem;
        color: #203b4c; line-height: 1.55;}
      .section-caption {color: #66788c; margin-top: -.35rem;}
      .muted {color: #77879a; font-size: .84rem;}
      @media (max-width: 760px) {
        .block-container {padding: 1rem .85rem 1.5rem;}
        .hero {padding: .9rem; align-items: flex-start;}
        .hero-icon svg {width: 25px; height: 25px;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="hero-icon" aria-hidden="true">
        <svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M16 3.5C16 3.5 7.7 13.1 7.7 19.2a8.3 8.3 0 0 0 16.6 0C24.3 13.1 16 3.5 16 3.5Z" stroke="currentColor" stroke-width="1.8"/>
          <path d="M11.5 20.1c.3 2.2 1.8 3.6 4.1 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <h1>AI-Based Flood Risk Assessment</h1>
        <div class="hero-subtitle">INDOFLOODS dataset &middot; Existing ML pipelines &middot; Historical severity classification</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="prototype-note">This application classifies the severity of recorded flood events. '
    'It is not a live sensor feed, a verified 24-hour forecast, or an operational emergency-warning system. Do not use it as the sole basis for real-world safety decisions.</div>',
    unsafe_allow_html=True,
)

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)


@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string, low_memory=False)


@st.cache_data(show_spinner=False)
def load_json(path_string: str):
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_pipeline(path_string: str):
    return joblib.load(path_string)


def pretty_model_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def class_label(value) -> str:
    text = str(value).strip().lower()
    if text in {"severe flood", "severe_flood", "severe-flood", "1", "1.0"}:
        return "Severe Flood"
    if text in {"flood", "0", "0.0"}:
        return "Flood"
    return str(value)


def status_item(label: str, value: str):
    icon_class = "status-missing" if value == "Missing" else "status-check"
    icon = "&#10007;" if value == "Missing" else "&#10003;"
    st.markdown(
        f'<div class="status-line"><span class="{icon_class}">{icon}</span>'
        f'<span>{label}: <strong>{value}</strong></span></div>',
        unsafe_allow_html=True,
    )


def probability_table(pipeline, row: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        probabilities = pipeline.predict_proba(row)[0]
        classifier = pipeline.named_steps.get("classifier") if hasattr(pipeline, "named_steps") else None
        classes = getattr(classifier, "classes_", range(len(probabilities)))
        return pd.DataFrame({
            "Class": [class_label(c) for c in classes],
            "Estimated model probability": probabilities.astype(float),
        })
    except Exception:
        return None


DATA_PATH = first_existing(DATA_CANDIDATES)
RESULTS_PATH = first_existing(RESULTS_CANDIDATES)
metadata = None
df = None
results = None
load_problems = []

if DATA_PATH:
    try:
        df = load_csv(str(DATA_PATH))
    except Exception as exc:
        load_problems.append(f"Could not read dataset `{DATA_PATH.relative_to(BASE_DIR)}`: {exc}")
if METADATA_PATH.exists():
    try:
        metadata = load_json(str(METADATA_PATH))
    except Exception as exc:
        load_problems.append(f"Could not read model metadata: {exc}")
if RESULTS_PATH:
    try:
        results = load_csv(str(RESULTS_PATH))
    except Exception as exc:
        load_problems.append(f"Could not read evaluation results: {exc}")

feature_columns = metadata.get("feature_names", []) if isinstance(metadata, dict) else []
model_names_from_metadata = metadata.get("models", []) if isinstance(metadata, dict) else []
model_names = [
    name for name in model_names_from_metadata
    if (MODEL_DIR / f"{name}.joblib").exists()
]
if (BASE_DIR / "outputs" / "ann" / "best_model.keras").exists():
    model_names.append("ann")
TUNED_MODEL_DIR = MODEL_DIR / "tuned"
model_names.extend(f"{p.stem}_tuned" for p in TUNED_MODEL_DIR.glob("*.joblib"))
if not model_names:
    model_names = sorted(p.stem for p in MODEL_DIR.glob("*.joblib") if p.stem != "best_model")

with st.sidebar:
    st.markdown("### Project status")
    st.caption("Member 3 - Dashboard and integration")
    status_item("Dataset", "Found" if DATA_PATH else "Missing")
    st.markdown(
        f'<div class="status-detail">{DATA_PATH.relative_to(BASE_DIR) if DATA_PATH else "Expected: data/processed/flood_model_data.csv"}</div>',
        unsafe_allow_html=True,
    )
    status_item("Training metadata", "Found" if METADATA_PATH.exists() else "Missing")
    status_item("Model pipelines found", str(len(model_names)))
    status_item("Evaluation results", "Found" if RESULTS_PATH else "Missing")
    st.markdown(
        f'<div class="status-detail">{RESULTS_PATH.relative_to(BASE_DIR) if RESULTS_PATH else "Expected: outputs/model_evaluation/model_comparison.csv"}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="side-target"><div class="side-target-label">Target</div>'
        '<div class="side-target-value">Flood Type</div>'
        '<div class="side-target-caption">Classes: Flood &middot; Severe Flood</div></div>',
        unsafe_allow_html=True,
    )

for problem in load_problems:
    st.error(problem)

if not DATA_PATH:
    st.error("Dataset file was not found. Expected `data/processed/flood_model_data.csv` in your project.")
if not METADATA_PATH.exists():
    st.error("`models/training_metadata.json` was not found. This file provides the exact feature names used during training.")
if not RESULTS_PATH:
    st.warning("Model comparison CSV was not found under `outputs/model_evaluation/` or `models/`.")
if not model_names:
    st.warning("No saved model pipelines (`*.joblib`) were found in the `models` folder.")

# The actual metric columns written by this project use snake_case.
metric_label_map = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
    "roc_auc": "ROC-AUC",
    "roc_auc_ovr_macro": "ROC-AUC (macro OVR)",
    "training_seconds": "Training time (seconds)",
}

tab_overview, tab_performance, tab_ann, tab_assessment = st.tabs([
    "Dataset Overview", "Model Performance", "Traditional ML vs Deep Learning", "Severity Assessment"
])

with tab_overview:
    st.subheader("Dataset overview")
    if df is None:
        st.info("Dataset statistics appear when `data/processed/flood_model_data.csv` is available.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("▤ Total records", f"{len(df):,}")
        c2.metric("▦ Dataset columns", f"{len(df.columns):,}")
        c3.metric("⌘ Model input features", f"{len(feature_columns):,}" if feature_columns else "Metadata unavailable")
        if "GaugeID" in df.columns:
            st.metric("Gauge stations", f"{df['GaugeID'].nunique():,}")
        elif "Station" in df.columns:
            st.metric("Stations", f"{df['Station'].nunique():,}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Flood-type distribution")
            if "Flood Type" in df.columns:
                counts = df["Flood Type"].fillna("Missing label").value_counts()
                ordered_labels = [
                    label for label in ["Flood", "Severe Flood", "Missing label"]
                    if label in counts.index
                ]
                other_labels = [label for label in counts.index if label not in ordered_labels]
                counts = counts.reindex(ordered_labels + other_labels)
                colors = [
                    "#3478b9" if label == "Flood" else
                    "#16a3a1" if label == "Severe Flood" else "#9aa8b7"
                    for label in counts.index
                ]
                fig, ax = plt.subplots(figsize=(7, 3.8))
                bars = ax.bar(counts.index, counts.values, color=colors, width=0.58)
                ax.set_ylabel("Recorded events", color="#42566d")
                ax.set_xlabel("Flood type", color="#42566d")
                ax.set_title("Recorded event counts", loc="left", color="#19344f", fontsize=12, pad=12)
                ax.yaxis.grid(True, color="#e5ebf1", linewidth=.8)
                ax.set_axisbelow(True)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.spines["bottom"].set_color("#cbd5df")
                ax.tick_params(axis="both", colors="#607287", length=0, pad=7)
                ax.bar_label(bars, labels=[f"{int(value):,}" for value in counts.values], padding=4, color="#263c52", fontsize=10)
                ax.margins(y=.16)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                st.dataframe(counts.rename("Events").rename_axis("Flood type").reset_index(), hide_index=True, width="stretch")
            else:
                st.warning("The `Flood Type` target column was not found.")
        with right:
            st.markdown("#### Data-quality summary")
            quality = pd.DataFrame({
                "Check": ["Rows", "Columns", "Duplicate full rows", "Missing target labels", "Features absent from dataset"],
                "Result": [
                    len(df), len(df.columns), int(df.duplicated().sum()),
                    int(df["Flood Type"].isna().sum()) if "Flood Type" in df.columns else "Target unavailable",
                    len([col for col in feature_columns if col not in df.columns]) if feature_columns else "Metadata unavailable",
                ],
            })
            st.dataframe(quality, hide_index=True, width="stretch")

        st.markdown("#### Sample records")
        preferred = [c for c in ["EventID", "GaugeID", "Station", "Start Date", "Flood Type"] if c in df.columns]
        st.dataframe(df[preferred].head(10) if preferred else df.head(10), hide_index=True, width="stretch")
        st.caption(f"Loaded from `{DATA_PATH.relative_to(BASE_DIR)}`.")

with tab_performance:
    st.subheader("Model comparison and evaluation")
    st.markdown('<div class="section-caption">Saved test-set results from the existing model experiment.</div>', unsafe_allow_html=True)
    if results is None or results.empty:
        st.info("Evaluation results appear after `outputs/model_evaluation/model_comparison.csv` is available.")
    else:
        result_display = results.copy()
        if "model" in result_display.columns and "Model" not in result_display.columns:
            result_display = result_display.rename(columns={"model": "Model"})
        st.dataframe(result_display.round(4), hide_index=True, width="stretch")

        metric_cols = [col for col in metric_label_map if col in results.columns]
        if "model" in results.columns and metric_cols:
            st.markdown("#### Compare evaluation metrics")
            chart = results.set_index("model")[metric_cols].apply(pd.to_numeric, errors="coerce")
            chart = chart.rename(columns={c: metric_label_map.get(c, c) for c in chart.columns})
            st.bar_chart(chart)

        if "f1_macro" in results.columns and "model" in results.columns:
            valid = results.copy()
            valid["f1_macro"] = pd.to_numeric(valid["f1_macro"], errors="coerce")
            valid = valid.dropna(subset=["f1_macro"])
            if not valid.empty:
                best_row = valid.loc[valid["f1_macro"].idxmax()]
                metric_cols = st.columns(3)
                metric_cols[0].metric("Highest macro F1 candidate", pretty_model_name(best_row["model"]))
                metric_cols[1].metric("Macro F1", f"{best_row['f1_macro']:.2%}")
                if "accuracy" in best_row.index and pd.notna(best_row["accuracy"]):
                    metric_cols[2].metric("Accuracy", f"{float(best_row['accuracy']):.2%}")
                st.caption("Highest macro F1 is a comparison result, not automatically the safest operational model. Review severe-flood recall and false negatives before choosing a model.")

    st.markdown("#### Saved evaluation plots")
    plot_specs = [
        ("accuracy_balanced_accuracy.png", "Accuracy and balanced accuracy"),
        ("macro_f1.png", "Macro F1 comparison"),
        ("per_class_recall.png", "Recall by class"),
        ("leading_confusion_matrices.png", "Confusion matrices for leading models"),
        ("roc_curves.png", "ROC curves"),
        ("precision_recall_curves.png", "Precision-recall curves"),
    ]
    available_plots = [(name, label) for name, label in plot_specs if (EVAL_DIR / name).exists()]
    if available_plots:
        for start in range(0, len(available_plots), 2):
            cols = st.columns(2)
            for col, (name, label) in zip(cols, available_plots[start:start + 2]):
                with col:
                    st.markdown(f"**{label}**")
                    st.image(str(EVAL_DIR / name), width="stretch")
    else:
        st.info("No saved PNG evaluation plots were found in `outputs/model_evaluation/`.")

    with st.expander("Presentation notes, limitations, and checklist"):
        st.markdown("**Suggested presentation flow**")
        st.markdown(
            "1. **Dataset Overview:** explain the historical event records and Flood Type classes.\n"
            "2. **Model Performance:** compare all eight saved models using the existing held-out results.\n"
            "3. **Evaluation plots:** explain macro F1, per-class recall, confusion matrices, ROC and precision-recall curves.\n"
            "4. **Severity Assessment:** select a historical record and run one of the saved pipelines.\n"
            "5. **Limitations:** describe false negatives and explain why this is not yet a live advance-warning service."
        )
        st.markdown("**Important limitations**")
        st.markdown(
            "- **Historical severity classification:** the target is `Flood Type` for recorded flood events, not flood/no-flood prediction for a future time window.\n"
            "- **False negatives:** a severe flood labelled as an ordinary flood can be dangerous; examine severe-class recall and confusion matrices.\n"
            "- **Macro metrics:** macro precision/recall/F1 average performance across classes and are not the same as severe-class recall alone.\n"
            "- **Historical-row demonstration:** predicting a row already in the dataset is for demonstrating the app and is not independent evaluation.\n"
            "- **Live early warning:** a real warning system needs time-stamped input available before the event, non-flood periods, a defined forecast horizon, live data integration, and independent validation."
        )
        st.markdown("**Member 3 checklist**")
        st.checkbox("Dashboard loads the existing dataset and metadata", key="m3_data")
        st.checkbox("All available model comparison metrics and plots display", key="m3_metrics")
        st.checkbox("A saved model pipeline successfully predicts a historical row", key="m3_predict")
        st.checkbox("Dashboard was tested and committed to GitHub", key="m3_git")

with tab_ann:
    st.subheader("Traditional ML vs Deep Learning")
    ann_dir = BASE_DIR / "outputs" / "ann"
    comparison_path = EVAL_DIR / "tuned" / "model_comparison.csv"
    if not comparison_path.exists():
        comparison_path = ann_dir / "model_comparison.csv"
    if not comparison_path.exists():
        st.info("ANN results are not available yet. Install requirements, then run `python -m src.ann.train` and `python -m src.ann.evaluate`.")
        st.caption("The ANN uses the existing stratified random test partition. The baseline does not use GaugeID grouping, so this comparison does not measure unseen-gauge generalisation.")
    else:
        comparison = load_csv(str(comparison_path))
        comparison["model_type"] = comparison["model"].map(lambda name: "Deep Learning" if name == "ann" else "Traditional ML")
        best = comparison.loc[pd.to_numeric(comparison["f1_macro"], errors="coerce").idxmax()]
        st.metric("Best macro F1 model", f"{pretty_model_name(best['model'])} · {best['model_type']}", f"{float(best['f1_macro']):.1%}")
        table_cols = [c for c in ["model", "model_type", "accuracy", "precision_macro", "recall_macro", "f1_macro"] if c in comparison]
        shown = comparison[table_cols].copy().rename(columns={"model": "Model", "model_type": "Type", "accuracy": "Accuracy", "precision_macro": "Macro precision", "recall_macro": "Macro recall", "f1_macro": "Macro F1"})
        for column in ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]:
            if column in shown: shown[column] = pd.to_numeric(shown[column], errors="coerce").map(lambda value: f"{value:.1%}" if pd.notna(value) else "—")
        st.dataframe(shown, hide_index=True, width="stretch")
        for metric, title in [("accuracy", "Accuracy"), ("f1_macro", "Macro F1-score"), ("Severe Flood_recall", "Severe Flood recall")]:
            if metric in comparison:
                st.markdown(f"#### {title}")
                chart = comparison[["model", "model_type", metric]].copy()
                chart[metric] = pd.to_numeric(chart[metric], errors="coerce") * 100
                try:
                    import plotly.express as px
                    fig = px.bar(chart, x="model", y=metric, color="model_type", text=chart[metric].map(lambda v: f"{v:.1f}%"),
                                 color_discrete_map={"Traditional ML": "#3478b9", "Deep Learning": "#16a3a1"}, labels={"model": "Model", metric: "Percent", "model_type": "Model type"})
                    fig.update_layout(yaxis_range=[0, 100], legend_title_text="Model type")
                    st.plotly_chart(fig, width="stretch")
                except ImportError:
                    st.bar_chart(chart.set_index("model")[metric])
        report_path = EVAL_DIR / "tuned" / "classification_reports.json"
        if not report_path.exists():
            report_path = ann_dir / "classification_reports.json"
        ann_metrics = load_json(str(ann_dir / "test_metrics.json")) if (ann_dir / "test_metrics.json").exists() else {}
        if report_path.exists() and "ann" in (reports := load_json(str(report_path))):
            ann_report = reports["ann"]
            old_models = [name for name in comparison.model if name != "ann"]
            best_old = comparison[comparison.model.isin(old_models)].sort_values("f1_macro", ascending=False).iloc[0]["model"]
            c1, c2 = st.columns(2)
            for col, name in [(c1, "ann"), (c2, best_old)]:
                report_source = load_json(str(report_path))
                rep = ann_report if name == "ann" else report_source[name]
                col.markdown(f"#### {pretty_model_name(name)} class-wise scores")
                col.dataframe(pd.DataFrame([{ "Class": label, "Precision": rep[label]["precision"], "Recall": rep[label]["recall"], "F1": rep[label]["f1-score"]} for label in ["Flood", "Severe Flood"]]).style.format({"Precision":"{:.1%}","Recall":"{:.1%}","F1":"{:.1%}"}), hide_index=True, width="stretch")
        for filename, title in [("training_accuracy.png", "ANN training and validation accuracy"), ("training_loss.png", "ANN training and validation loss"), ("confusion_matrix.png", "ANN confusion matrix")]:
            if (ann_dir / filename).exists(): st.image(str(ann_dir / filename), caption=title, width="stretch")
        if (ann_dir / "training_history.csv").exists():
            history = load_csv(str(ann_dir / "training_history.csv"))
            if "loss" in history and "val_loss" in history:
                gap = float(history.iloc[-1]["val_loss"] - history.iloc[-1]["loss"])
                overfit_text = "The final validation loss exceeds training loss, which is a possible overfitting signal." if gap > 0.1 else "The final training and validation losses are close; this alone does not establish generalisation."
            else: overfit_text = "Training history is unavailable."
            ann_row = comparison[comparison.model == "ann"].iloc[0]
            top_old = comparison[comparison.model != "ann"].sort_values("f1_macro", ascending=False).iloc[0]
            text = f"ANN macro F1 is {float(ann_row.f1_macro):.1%}; the best traditional model ({pretty_model_name(top_old.model)}) is {float(top_old.f1_macro):.1%}. "
            text += "ANN improves macro F1 on this test set." if ann_row.f1_macro > top_old.f1_macro else "ANN does not improve macro F1 on this test set."
            text += f" Severe Flood recall is {float(ann_row['Severe Flood_recall']):.1%} for ANN and {float(top_old['Severe Flood_recall']):.1%} for the best traditional model. {overfit_text}"
            st.markdown("#### Interpretation")
            st.write(text)
            st.caption("Interpretation is calculated from saved held-out metrics and training history. Results are conditional on the baseline random split; gauge-wise generalisation was not evaluated.")

with tab_assessment:
    st.subheader("Assess a recorded event")
    st.markdown('<div class="section-caption">Select a historical row and an existing saved pipeline.</div>', unsafe_allow_html=True)
    if df is None or not feature_columns:
        st.info("This section needs the dataset and `models/training_metadata.json`.")
    elif not model_names:
        st.info("No saved pipelines were detected in the `models` folder.")
    elif "Flood Type" not in df.columns:
        st.error("The dataset must contain the `Flood Type` target column for this demonstration.")
    else:
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            st.error(f"The dataset is missing {len(missing_features)} model features, for example: {', '.join(missing_features[:8])}")
        else:
            model_for_display = {pretty_model_name(name): name for name in model_names}
            result_best = None
            if results is not None and "model" in results.columns and "f1_macro" in results.columns:
                temp = results.copy()
                temp["f1_macro"] = pd.to_numeric(temp["f1_macro"], errors="coerce")
                temp = temp.dropna(subset=["f1_macro"])
                if not temp.empty:
                    result_best = str(temp.loc[temp["f1_macro"].idxmax(), "model"])
            default_name = result_best if result_best in model_names else ("gradient_boosting" if "gradient_boosting" in model_names else model_names[0])
            if result_best and (TUNED_MODEL_DIR / f"{result_best}.joblib").exists():
                default_name = f"{result_best}_tuned"
            shown_models = list(model_for_display.keys())
            default_display = pretty_model_name(default_name)
            selected_display = st.selectbox("Choose trained model", shown_models, index=shown_models.index(default_display))
            selected_model_name = model_for_display[selected_display]

            def event_text(i):
                row = df.iloc[i]
                ident = row.get("EventID", row.get("GaugeID", f"Dataset row {i + 1}"))
                actual = row.get("Flood Type", "Unknown")
                return f"Row {i + 1} | {ident} | recorded label: {actual}"

            selected_pos = st.selectbox("Select a historical event", list(range(len(df))), format_func=event_text)
            selected_row = df.iloc[[selected_pos]]
            X_input = selected_row.reindex(columns=feature_columns)
            with st.expander("View the input features provided to the pipeline"):
                st.dataframe(X_input.T.astype("string").rename(columns={X_input.index[0]: "Value"}), width="stretch")
                st.caption("The saved training pipeline handles the imputation, scaling and categorical encoding used during training.")

            if st.button("Estimate severity", type="primary", width="stretch"):
                try:
                    if selected_model_name == "ann":
                        import tensorflow as tf
                        ann_dir = BASE_DIR / "outputs" / "ann"
                        ann_meta = load_pipeline(str(ann_dir / "metadata.joblib"))
                        ann_preprocessor = load_pipeline(str(ann_dir / "preprocessor.joblib"))
                        ann_model = tf.keras.models.load_model(ann_dir / "best_model.keras")
                        predicted, probability = predict_ann(ann_model, ann_preprocessor, X_input, ann_meta["feature_names"])
                        probs = pd.DataFrame({"Class": ["Flood", "Severe Flood"], "Estimated model probability": [1 - probability, probability]})
                    else:
                        if selected_model_name.endswith("_tuned"):
                            tuned_name = selected_model_name.removesuffix("_tuned")
                            pipeline = load_pipeline(str(TUNED_MODEL_DIR / f"{tuned_name}.joblib"))
                        else:
                            pipeline = load_pipeline(str(MODEL_DIR / f"{selected_model_name}.joblib"))
                        predicted = pipeline.predict(X_input)[0]
                        probs = probability_table(pipeline, X_input)
                    predicted_text = class_label(predicted)
                    predicted_probability = None
                    if probs is not None:
                        matched = probs.loc[probs["Class"] == predicted_text]
                        if not matched.empty:
                            predicted_probability = float(matched.iloc[0]["Estimated model probability"])

                    heading = "LEVEL 2 - SEVERE FLOOD CLASS" if predicted_text == "Severe Flood" else "LEVEL 1 - FLOOD CLASS"
                    explanation = (
                        "The selected pipeline classified this historical record as Severe Flood. Review false negatives and the model limitations."
                        if predicted_text == "Severe Flood" else
                        "The selected pipeline classified this historical record as Flood. This does not mean the location is safe or that a future flood cannot occur."
                    )
                    st.markdown(f'<div class="prediction"><strong>{heading}</strong><br>{explanation}</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted class", predicted_text)
                    c2.metric("Model probability for predicted class", f"{predicted_probability:.1%}" if predicted_probability is not None else "Not available")
                    if probs is not None:
                        st.markdown("#### Class probabilities")
                        st.dataframe(probs.assign(**{"Estimated model probability": probs["Estimated model probability"].map(lambda x: f"{x:.1%}")}), hide_index=True, width="stretch")
                    st.caption(f"Recorded historical label: **{selected_row['Flood Type'].iloc[0]}**. This demonstrates pipeline operation; it is not an independent forecast test.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.info("Confirm that the dataset is `data/processed/flood_model_data.csv` and that the selected `.joblib` file was created from the same training metadata/features.")

st.divider()
st.markdown('<div class="muted">Academic prototype | Evaluation values are taken from saved test results. Not an official flood warning service.</div>', unsafe_allow_html=True)
