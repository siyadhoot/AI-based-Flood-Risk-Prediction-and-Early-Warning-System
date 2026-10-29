# Member 3 — Dashboard and Integration

This package provides the Streamlit dashboard for the **AI-Based Flood Risk Prediction and Early Warning System** course project.

## 1. Files expected from the other team members

Place these files in the same project root as `app.py`:

```text
flood-risk-prediction/
├── app.py
├── requirements.txt
├── data/
│   └── combined_flood_dataset.csv
└── models/
    ├── best_model.joblib
    ├── feature_columns.json
    └── model_comparison.csv
```

- **Member 1** prepares and saves `data/combined_flood_dataset.csv`.
- **Member 2** trains/evaluates the model and saves the three files in `models/`.
- **Member 3** provides `app.py`, connects the saved pipeline to the interface, checks the outputs, and demonstrates the system.

The dashboard intentionally shows clear messages when a file is missing; it does not generate fake metrics or model predictions.

## 2. Run it on Windows

Open Command Prompt or the VS Code terminal in the project root and run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If the environment already exists, skip the first two commands. Streamlit normally opens the dashboard in your browser.

## 3. Dashboard sections

- **Dataset Overview:** event/station counts, target distribution, sample records, and basic data-quality checks.
- **Model Performance:** actual comparison values loaded from `model_comparison.csv`, charts for available metrics, and an F1-score candidate.
- **Severity Assessment:** choose a recorded event, run the saved scikit-learn pipeline, and inspect the predicted class and class probabilities if supported.
- **Demo & Limitations:** a quick presentation sequence and the main safety/methodology limitations.

## 4. Important scope statement

The current target, `Flood Type`, distinguishes `Flood` from `Severe Flood` among historical recorded flood events. Therefore, the current app demonstrates **historical flood severity classification**. It must not be described as a proven live forecast or a verified 24-hour warning system. Genuine early warning requires a prediction-time dataset with both flood and non-flood periods, a clearly defined future horizon, input data available before the outcome, and independent validation.

The dashboard's Level 1 / Level 2 labels are explanatory labels for the two model classes, **not official government alert levels**. A probability from a classifier is not necessarily a calibrated real-world probability.

## 5. Fast integration checks

1. Confirm the four required files appear as found in the sidebar.
2. Confirm the Dataset Overview count matches the combined dataset.
3. Confirm model names and metrics match Member 2's CSV.
4. Select a historical event and click **Estimate severity**.
5. Discuss false negatives, severe-class recall, false alarms, unseen-station generalisation, missing data, and why this is not yet live early warning.

## 6. Push the contribution to GitHub

Run these commands from the existing cloned repository. Replace `main` if your team uses another branch.

```bash
git status
git pull --rebase origin main
git add app.py requirements.txt MEMBER3_README.md .gitignore
git add data/combined_flood_dataset.csv models/best_model.joblib models/feature_columns.json models/model_comparison.csv
git status
git commit -m "Add Streamlit flood severity dashboard"
git push origin main
```

If your teammate's dataset/model files are not in place yet, stage only the files you currently have and commit the dashboard shell separately. Before pushing, check the staged files with `git status` and make sure you are not committing secrets, virtual environments, or unrelated files.
