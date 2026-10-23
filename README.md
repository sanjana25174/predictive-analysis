# AI-Based Smart Manufacturing & Predictive Maintenance System

A Streamlit dashboard on top of your trained Random Forest. It estimates machine-failure probability from five sensor
readings and a machine type, maps that probability to a prototype risk level, and shows a maintenance recommendation.

> **Prototype for learning and demonstration.** It uses the public AI4I 2020 dataset, needs manual sensor input, and has
> not been validated on any real machine. It is a decision-support estimate, not a safety system.

---

## 1. Setup and run (Windows PowerShell, Python 3.12)

Open PowerShell and run these commands one at a time.

```powershell
cd "C:\Users\raksh\OneDrive\Desktop\Documents\AI_Predictive_Maintenance"

# Create a virtual environment OUTSIDE OneDrive (keeps OneDrive from syncing thousands of library files)
python -m venv "$env:USERPROFILE\venvs\ai_pm"
& "$env:USERPROFILE\venvs\ai_pm\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install -r requirements.txt

# Regenerate the model-comparison metrics with YOUR installed versions (takes a few seconds)
python evaluate_models.py

# Run the automated logic tests
python test_core.py

# Launch the dashboard
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>. Stop it with `Ctrl + C` in PowerShell.

- If PowerShell says *running scripts is disabled* when activating the environment, run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and activate again.
- If Streamlit asks for an email on first start, just press Enter.
- If you already have an environment where Streamlit works, you can skip the `venv` lines and only run
  `pip install -r requirements.txt` (the new dependency is `plotly`).
- Later, open the app with: activate the environment, `cd` into the project folder, `streamlit run app.py`.

---

## 2. What each file does

| File | Status | Purpose |
|---|---|---|
| `app.py` | **replaced** | The dashboard (all 7 pages). Your original is saved as `legacy/app_v1.py`. |
| `core.py` | **new** | Loading, encoding, validation, prediction, risk levels and evaluation, with no Streamlit code so it can be tested. |
| `evaluate_models.py` | **new** | Creates `model_metrics.json`. See "Why this file exists" below. |
| `model_metrics.json` | **new (generated)** | Test and train metrics for all three models. Re-create it with `python evaluate_models.py`. |
| `test_core.py` | **new** | 16 automated checks of the logic layer. |
| `requirements.txt` | **updated** | Adds `plotly`, `streamlit>=1.40`, `scikit-learn>=1.9.1`. |
| `.streamlit/config.toml` | **new** | Light theme and teal accent. |
| `legacy/app_v1.py` | **new (backup)** | Your original app, kept for reference. Delete it if you do not need it. |
| `data_cleaning.py`, `model_training.py` | unchanged | Training-time pipeline, preserved exactly. |
| `prediction_system.py` | unchanged | Original command-line prototype. Its risk bands (30% / 70%) are the dashboard's defaults. |
| `ai4i2020.csv`, `ai4i2020_cleaned.csv`, `random_forest_model.pkl`, `feature_names.pkl` | unchanged | Your data and trained model. |

**Why `model_training.py` was not changed.** The saved model reproduces your reported results exactly, so there was no
reason to retrain it. Retraining would risk silently changing the model you already reported.

**Why `evaluate_models.py` exists.** `model_training.py` only *printed* metrics, and it saved only the Random Forest.
Logistic Regression and Decision Tree numbers were therefore not stored anywhere. The script:
- loads your **saved** Random Forest (it is *not* retrained) and evaluates it on the held-out test split;
- retrains Logistic Regression and Decision Tree with the identical settings, split and `random_state` from
  `model_training.py` (they are never used for predictions) and evaluates them;
- checks the Random Forest against your reported figures and prints `VERIFIED` or `MISMATCH`.

---

## 3. How the prediction works

1. The user enters five sensor values and a machine type; inputs are validated (missing, non-numeric, infinite, or
   physically impossible values are rejected; values outside the dataset's observed range only trigger a warning).
2. `core.build_model_input` one-hot encodes the machine type into `Type_H`, `Type_L`, `Type_M` and orders the columns
   exactly like `feature_names.pkl`.
3. The **saved** Random Forest returns the predicted class (`predict`) and failure probability (`predict_proba`).
   Nothing is random or hardcoded.
4. The probability is mapped to a risk level with configurable **prototype** bands (default: below 30% lower,
   30% to below 70% medium, 70% and above higher). These are not industrially validated safety limits.
5. The predicted class uses the model's own 50% rule, so a prediction can be "No failure predicted" with a medium risk
   level (for example 40%). The result panel explains this.

---

## 4. Verified results

Re-checked against your files: the saved model on the reproduced 20% stratified test split (2,000 records, 68 failures)
gives accuracy 96.90%, precision 53.12%, recall 75.00%, F1 62.20%, and confusion matrix TN 1887, FP 45, FN 17, TP 51.
This matches what you reported.

Recomputed comparison models (test set):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 82.50% | 14.21% | 82.35% | 24.24% |
| Decision Tree | 92.95% | 31.09% | 88.24% | 45.98% |
| Random Forest | 96.90% | 53.12% | 75.00% | 62.20% |

Be ready to explain this in an interview: the Random Forest has the best accuracy, precision and F1-score, but the
Decision Tree and Logistic Regression have *higher recall*; they catch more real failures at the cost of many more false
alarms. The Random Forest also fits the training data much more closely than the test data (training recall about 99.6%
vs 75% on test), which the "overfitting check" panel on the Model Performance page shows.

---

## 5. Test checklist

### Automated (run these in PowerShell)
- [ ] `python test_core.py` prints `16 of 16 tests passed.`
- [ ] `python evaluate_models.py` ends with `VERIFIED: the saved Random Forest reproduces every reported figure.`

### Manual (open the dashboard and tick each box)

**Navigation**
- [ ] All 7 sidebar items open a different page, and the active item is highlighted.
- [ ] The sidebar and workspace look right at a narrow browser width (columns stack).

**Overview Dashboard**
- [ ] Title is "Smart Manufacturing Dashboard" with the subtitle and today's date at the top right.
- [ ] KPI cards show 10,000 records, 339 failures, 9,661 non-failures, and 96.90% accuracy.
- [ ] The blue banner says these are historical dataset statistics, not live machines.
- [ ] The failure-distribution chart, sensor table, five histograms and machine-type chart appear; hovering shows values.
- [ ] Model summary shows 96.90 / 53.12 / 75.00 / 62.20 with the accuracy-vs-imbalance explanation.
- [ ] "Open Predict Machine Failure" opens the Predict page.

**Predict Machine Failure**
- [ ] Defaults are filled in and no warnings show.
- [ ] Click Predict: a gauge, a risk badge, probability, predicted class, recommendation and entered values appear.
- [ ] Open "Show the encoded row sent to the model": 8 columns in the order `Air...`, `Process...`, `Rotational...`,
      `Torque...`, `Tool wear...`, `Type_H`, `Type_L`, `Type_M`, with exactly one machine-type column equal to 1.
- [ ] Changing machine type from L to H changes the one-hot columns; click Predict again to see the new result.
- [ ] Enter torque `120`: an "outside the observed dataset range" warning appears, but prediction still works.
- [ ] Each of the four sample buttons fills the form; Predict then shows whatever the model returns.
- [ ] After a prediction, change any input: "The inputs have changed since this prediction" appears.
- [ ] In "Risk thresholds", set "Medium risk starts at" above "Higher risk starts at": an error appears and the
      Predict button is disabled. "Reset to project defaults" restores 0.30 / 0.70.
- [ ] Go to another page and come back: your inputs and last result are still there.

**Machine Health**
- [ ] Banner says the analysis is historical and not connected to machines.
- [ ] Summary table, group-means table, standardized-difference chart appear.
- [ ] Changing the sensor selector updates the histogram, box plot and failure-rate chart; machine-type chart appears.

**Model Performance**
- [ ] Verification table shows a green "Verified" message with every row marked "✔ Yes".
- [ ] Comparison chart shows three models x four metrics, and the status table says which values were verified and
      which were recomputed.
- [ ] Confusion matrix shows 1,887 / 45 / 17 / 51 with plain-English explanations.
- [ ] Moving the threshold slider changes precision, recall, false alarms and missed failures.
- [ ] The risk-band table's three "Test records" values add up to 2,000.
- [ ] Temporarily rename `model_metrics.json`: a warning appears telling you to run `python evaluate_models.py`
      (rename it back afterwards).

**Sensor Analytics**
- [ ] Machine type, outcome and tool-wear filters change the record count caption and every chart.
- [ ] Five tabs each show a histogram and box plot; the scatter plot axes can be changed; heatmap and
      failure-vs-non-failure box plots appear.
- [ ] Deselecting every machine type shows "No records match" instead of an error.

**Prediction History**
- [ ] Before any prediction, an empty message with a "Go to Predict Machine Failure" button appears.
- [ ] After predictions, every one appears with timestamp, inputs, class, probability, risk level, recommendation.
- [ ] "Download history as CSV" works. "Clear history" asks for confirmation; Cancel keeps records; Yes deletes them.
- [ ] Refreshing the browser page (F5) empties the history, as the banner says.

**About Project**
- [ ] Problem, solution, dataset, stack, models, two-lane architecture (training time vs prediction time),
      limitations, future scope and the system-status table all appear. The status table shows the scikit-learn
      version the model was saved with versus the one running.

---

## 6. Demo script (about 5 minutes)

1. **Overview** (30 s): "These numbers are the historical AI4I dataset, not live machines. It is only 3.39% failures, so
   accuracy alone is misleading; a model that always says 'no failure' scores about 96.6%."
2. **Model Performance** (90 s): show the verification table, then the comparison chart. Point out that Random Forest wins
   on precision/F1 while Decision Tree has higher recall, and explain false alarms vs missed failures from the confusion
   matrix. Move the threshold slider to show the trade-off.
3. **Predict** (90 s): load "Typical readings" and predict; load "High torque and wear" and predict; open the encoded-row
   expander to show feature order. Say that the probability comes from `predict_proba` of the saved model, and the risk
   bands are prototype settings you can change.
4. **Prediction History** (20 s): show the two predictions and the confirm-before-clear behavior. Mention it is temporary.
5. **Machine Health / Sensor Analytics** (60 s): pick Torque, show the failure-rate-by-bin chart, then the correlation
   heatmap. Say correlation is not causation.
6. **About** (20 s): show the training-time vs prediction-time architecture and the limitations.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: plotly` (or streamlit) | Activate your environment, then `pip install -r requirements.txt`. |
| Red box "file was not found" | Run `streamlit run app.py` from inside the project folder; keep all files together. |
| Warning about scikit-learn versions on the About page | `pip install scikit-learn==1.9.1`, or re-run `model_training.py` then `python evaluate_models.py`. |
| You retrained the model and the dashboard still shows old numbers | Run `python evaluate_models.py`, then in the browser press `C` and choose "Clear cache", or restart Streamlit. |
| Sidebar or cards look plain | Upgrade Streamlit: `pip install --upgrade streamlit`. |

Only load `.pkl` files you created yourself or trust; unpickling a file can run code.

---

## 8. Honest limits of this prototype

- The dataset is public and simulated industrial data. Results do not transfer automatically to real factory machines.
- Inputs are typed by hand; there is no live sensor connection.
- Risk bands are prototype settings. The "risk band" table on the Model Performance page shows how they behave on this
  dataset, but that is not validation for real use.
- The model produces false alarms and misses some failures (see the confusion matrix).
