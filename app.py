from pathlib import Path
import json
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "processed" / "flood_model_data.csv",
    BASE_DIR / "data" / "processed" / "flood_cleaned.csv",
    BASE_DIR / "data" / "processed" / "flood_combined.csv",
    BASE_DIR / "data" / "combined_flood_dataset.csv",
]
METADATA_PATH = BASE_DIR / "models" / "training_metadata.json"
RESULTS_CANDIDATES = [
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
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      .hero {padding: 1.2rem 1.4rem; border-radius: 14px;
        background: linear-gradient(120deg, rgba(15,76,117,.16), rgba(50,130,184,.08));
        border: 1px solid rgba(100,140,170,.30); margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .35rem 0;}
      .muted {color: #77808a; font-size: .92rem;}
      .prediction {padding: .9rem 1rem; border-radius: 10px;
        border: 1px solid rgba(130,130,130,.35); margin: .5rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <h1>🌊 AI-Based Flood Risk Assessment</h1>
      <div>INDOFLOODS dataset · Existing ML pipelines · Historical severity classification</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Prototype only: this app classifies the severity of recorded flood events. "
    "It is not a live sensor feed, a verified 24-hour forecast, or an operational emergency-warning system. "
    "Do not use it as the sole basis for real-world safety decisions."
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
if not model_names:
    model_names = sorted(p.stem for p in MODEL_DIR.glob("*.joblib") if p.stem != "best_model")

with st.sidebar:
    st.header("Project status")
    st.caption("Member 3 · Dashboard and integration")
    st.write(f"Dataset: {'✅ Found' if DATA_PATH else '❌ Missing'}")
    st.caption(str(DATA_PATH.relative_to(BASE_DIR)) if DATA_PATH else "Expected: data/processed/flood_model_data.csv")
    st.write(f"Training metadata: {'✅ Found' if METADATA_PATH.exists() else '❌ Missing'}")
    st.write(f"Model pipelines found: {len(model_names)}")
    st.write(f"Evaluation results: {'✅ Found' if RESULTS_PATH else '❌ Missing'}")
    st.caption(str(RESULTS_PATH.relative_to(BASE_DIR)) if RESULTS_PATH else "Expected: outputs/model_evaluation/model_comparison.csv")
    st.divider()
    st.markdown("**Target**")
    st.write("Flood Type")
    st.write("Classes: Flood · Severe Flood")

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

tab_overview, tab_performance, tab_assessment, tab_demo = st.tabs([
    "📊 Dataset Overview", "🧪 Model Performance", "🚦 Severity Assessment", "🎤 Demo & Limitations"
])

with tab_overview:
    st.subheader("Dataset overview")
    if df is None:
        st.info("Dataset statistics appear when `data/processed/flood_model_data.csv` is available.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Dataset columns", f"{len(df.columns):,}")
        c3.metric("Model input features", f"{len(feature_columns):,}" if feature_columns else "Metadata unavailable")
        if "GaugeID" in df.columns:
            st.metric("Gauge stations", f"{df['GaugeID'].nunique():,}")
        elif "Station" in df.columns:
            st.metric("Stations", f"{df['Station'].nunique():,}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Flood-type distribution")
            if "Flood Type" in df.columns:
                counts = df["Flood Type"].fillna("Missing label").value_counts().rename_axis("Flood type")
                st.bar_chart(counts)
                st.dataframe(counts.rename("Events").reset_index(), hide_index=True, use_container_width=True)
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
            st.dataframe(quality, hide_index=True, use_container_width=True)

        st.markdown("#### Sample records")
        preferred = [c for c in ["EventID", "GaugeID", "Station", "Start Date", "Flood Type"] if c in df.columns]
        st.dataframe(df[preferred].head(10) if preferred else df.head(10), hide_index=True, use_container_width=True)
        st.caption(f"Loaded from `{DATA_PATH.relative_to(BASE_DIR)}`.")

with tab_performance:
    st.subheader("Model comparison and evaluation")
    st.caption("The table below uses your existing held-out evaluation results.")
    if results is None or results.empty:
        st.info("Evaluation results appear after `outputs/model_evaluation/model_comparison.csv` is available.")
    else:
        result_display = results.copy()
        if "model" in result_display.columns and "Model" not in result_display.columns:
            result_display = result_display.rename(columns={"model": "Model"})
        st.dataframe(result_display.round(4), hide_index=True, use_container_width=True)

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
                    st.image(str(EVAL_DIR / name), use_container_width=True)
    else:
        st.info("No saved PNG evaluation plots were found in `outputs/model_evaluation/`.")

with tab_assessment:
    st.subheader("Assess a recorded event")
    st.caption("Select a historical row and one of the pipelines already saved by your team's training script.")
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
            shown_models = list(model_for_display.keys())
            default_display = pretty_model_name(default_name)
            selected_display = st.selectbox("Choose trained model", shown_models, index=shown_models.index(default_display))
            selected_model_name = model_for_display[selected_display]

            def event_text(i):
                row = df.iloc[i]
                ident = row.get("EventID", row.get("GaugeID", f"Dataset row {i + 1}"))
                actual = row.get("Flood Type", "Unknown")
                return f"Row {i + 1} · {ident} · recorded label: {actual}"

            selected_pos = st.selectbox("Select a historical event", list(range(len(df))), format_func=event_text)
            selected_row = df.iloc[[selected_pos]]
            X_input = selected_row.reindex(columns=feature_columns)
            with st.expander("View the input features provided to the pipeline"):
                st.dataframe(X_input.T.rename(columns={X_input.index[0]: "Value"}), use_container_width=True)
                st.caption("The saved training pipeline handles the imputation, scaling and categorical encoding used during training.")

            if st.button("Estimate severity", type="primary", use_container_width=True):
                model_path = MODEL_DIR / f"{selected_model_name}.joblib"
                try:
                    pipeline = load_pipeline(str(model_path))
                    predicted = pipeline.predict(X_input)[0]
                    predicted_text = class_label(predicted)
                    probs = probability_table(pipeline, X_input)
                    predicted_probability = None
                    if probs is not None:
                        matched = probs.loc[probs["Class"] == predicted_text]
                        if not matched.empty:
                            predicted_probability = float(matched.iloc[0]["Estimated model probability"])

                    heading = "LEVEL 2 · SEVERE FLOOD CLASS" if predicted_text == "Severe Flood" else "LEVEL 1 · FLOOD CLASS"
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
                        st.dataframe(probs.assign(**{"Estimated model probability": probs["Estimated model probability"].map(lambda x: f"{x:.1%}")}), hide_index=True, use_container_width=True)
                    st.caption(f"Recorded historical label: **{selected_row['Flood Type'].iloc[0]}**. This demonstrates pipeline operation; it is not an independent forecast test.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.info("Confirm that the dataset is `data/processed/flood_model_data.csv` and that the selected `.joblib` file was created from the same training metadata/features.")

with tab_demo:
    st.subheader("Suggested presentation flow")
    st.markdown(
        "1. **Dataset Overview:** explain the historical event records and Flood Type classes.\n"
        "2. **Model Performance:** compare all eight saved models using the existing held-out results.\n"
        "3. **Evaluation plots:** explain macro F1, per-class recall, confusion matrices, ROC and precision-recall curves.\n"
        "4. **Severity Assessment:** select a historical record and run one of the saved pipelines.\n"
        "5. **Limitations:** describe false negatives and explain why this is not yet a live advance-warning service."
    )
    st.markdown("#### Important limitations")
    st.markdown(
        "- **Historical severity classification:** the target is `Flood Type` for recorded flood events, not flood/no-flood prediction for a future time window.\n"
        "- **False negatives:** a severe flood labelled as an ordinary flood can be dangerous; examine severe-class recall and confusion matrices.\n"
        "- **Macro metrics:** macro precision/recall/F1 average performance across classes and are not the same as severe-class recall alone.\n"
        "- **Historical-row demonstration:** predicting a row already in the dataset is for demonstrating the app and is not independent evaluation.\n"
        "- **Live early warning:** a real warning system needs time-stamped input available before the event, non-flood periods, a defined forecast horizon, live data integration, and independent validation."
    )
    st.markdown("#### Member 3 checklist")
    st.checkbox("Dashboard loads the existing dataset and metadata", key="m3_data")
    st.checkbox("All available model comparison metrics and plots display", key="m3_metrics")
    st.checkbox("A saved model pipeline successfully predicts a historical row", key="m3_predict")
    st.checkbox("Dashboard was tested and committed to GitHub", key="m3_git")

st.divider()
st.markdown('<div class="muted">Academic prototype · Evaluation values are taken from saved test results. Not an official flood warning service.</div>', unsafe_allow_html=True)
