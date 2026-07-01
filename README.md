# Psychometric Predictor

Machine-learning analysis that classifies DASS-21 depression, anxiety, and stress severity from survey responses collected during COVID-19 lockdown in India.

## Problem Statement & Goals

The notebook builds supervised classifiers to predict ordinal mental-health severity labels (`depression_level`, `anxiety_level`, `stress_level`) from questionnaire and contextual inputs. The workflow also:

- Tests whether economic and demographic variables improve prediction beyond DASS-21 items alone.
- Evaluates models with stratified hold-out tests and cross-validation.
- Uses permutation importance to check whether predictive signal aligns with the DASS-21 subscale structure (7 depression, 7 anxiety, 7 stress items).
- Visualizes decision boundaries and confusion patterns between adjacent severity classes.

⚠️ This is an exploratory psychometric / ML study, not a **clinical diagnostic tool.**

## Dataset

| Property | Detail |
|---|---|
| **Source** | [Mendeley Research Data](https://data.mendeley.com/) |
| **Size** | 978 respondents × 93 columns (87 after dropping sensitive/unused fields) |
| **Context** | Indian metro respondents (e.g. Kolkata, Mumbai, Delhi); COVID lockdown-era survey with **T1** and **T2** timepoints |
| **DASS-21 items (21 features)** | Ordinal Likert responses (4 levels: *Did not apply* → *Applied very much or most of the time*) for items such as `felt_downhearted_and_blue`, `dry_mouth`, `finding_hard_to_wind_down`, etc. |
| **Contextual / socioeconomic fields** | `family_economicaly_affected`, `education_of_children`, `covid_case`, `monthly_expenditure`, `leisure_time`, employment, housing, family composition, and related columns |
| **Demographics (retained but mostly unused in final models)** | `city`, `age`, `sex`, `study_level`, `occupation`, marital/residential patterns |
| **Targets** | `depression_level`, `anxiety_level`, `stress_level` — five classes: **Normal**, **Mild**, **Moderate**, **Severe**, **Extremely severe** |
| **Derived scores (not used as model inputs)** | Raw subscale totals (`depression T1`, `anxiety T1`, `stress T1`), `dass21_raw T1`, `zdass21 T1` |

After preprocessing, the notebook splits the wide table into:

- `mh_T1.csv` — 978 × 51 (lockdown-timepoint features and T1 outcomes)
- `mh_T2.csv` — 978 × 49 (follow-up timepoint; exported but not modeled)

Dropped columns for privacy / redundancy: `name`, `religion`, `caste`, `owned/rented`, `mpce T1`, `mpce T2`, and relative-change fields (`relative_dep`, `relative_anx`, `relative_stress`).

## Methodology

### Preprocessing

1. Load `adult mental health in indian metro cities 2001-2021(cleaned).csv` from repo.
2. Drop identifying and unused columns; fix one column naming inconsistency (`finding_hard_to_wind_down_T1` → `finding_hard_to_wind_down T1`).
3. Split T1 / T2 suffixed columns into separate frames; strip suffixes.
4. Impute missing `education_of_children` with `'Unknown'`.
5. **Feature encoding**
   - **21 DASS items**: `OrdinalEncoder` with fixed 4-level category order.
   - **Extra categoricals** (in early experiments): `OneHotEncoder(drop='first')` for `family_economicaly_affected`, `education_of_children`, `covid_case`.
6. **Scaling**: `StandardScaler` on encoded features.
7. **Train/test split**: 80/20, stratified by target (`782` train / `196` test).

### Models

- **Algorithm**: `LogisticRegression` (`class_weight='balanced'`, `max_iter=3000`, `random_state=42`).
- **Primary feature set (final)**: all 21 DASS-21 ordinal items only (`ordinal_cols`).
- **Earlier experiments**: added economic / COVID context features, then removed them after ablation showed little or negative contribution.

### Evaluation

- Hold-out **accuracy**, **weighted F1**, **classification report**, and **confusion matrix** heatmaps.
- **Stratified K-fold cross-validation** (15-fold for depression refinement, 20-fold for anxiety/stress) on training data.
- **Permutation importance** (`n_repeats=30`, weighted F1 scoring) to rank features.
- **PCA (2 components)** + retrained logistic model for 2D decision-boundary visualization (depression).
- **Incremental feature evaluation** at the end: adds demographic/socioeconomic fields one at a time and reports whether performance improves.

### Feature-selection narrative in the notebook

1. Start with DASS-21 + economic/COVID features → moderate performance.
2. Remove economic factors → slight improvement.
3. Remove `education_of_children` and `covid_case` → best depression CV F1.
4. Reuse the 21-item set for anxiety and stress models.
5. Attempt to add demographics → accuracy and F1 decrease.

## Results & Key Findings

### Final models (21 DASS-21 items only)

| Outcome | Test accuracy | Test weighted F1 | CV mean weighted F1 (± std) |
|---|---:|---:|---|
| **Depression** | 0.908 | 0.911 | 0.910 ± 0.052 |
| **Anxiety** | 0.89 | 0.901 | 0.906 ± 0.033 |
| **Stress** | 0.916 | 0.940 | 0.920 ± 0.019 |

### Interpretation highlights

- **Misclassifications are mostly between adjacent severity bands** (e.g. Normal ↔ Mild, Severe ↔ Extremely severe), suggesting the model captures ordered structure even though severity is treated as nominal classes.
- **Permutation importance aligns with DASS-21 theory**: top contributors for each outcome map to that subscale’s 7 items (depression items for depression level, anxiety items for anxiety level, stress items for stress level).
- **Economic and demographic context did not help** the final models; the 21 self-report items alone are the strongest predictors in this dataset.
- **Class imbalance** is present (e.g. depression test set: 99 Normal vs 19 Mild); `class_weight='balanced'` mitigates but minority classes (especially Mild) remain weaker.

### Reference literature (linked in notebook)

- Psychometric validation papers on the 21-item DASS scale ([Nature](https://www.nature.com/articles/s41599-022-01229-x), [PMC8507889](https://pmc.ncbi.nlm.nih.gov/articles/PMC8507889/), [PMC10670895](https://pmc.ncbi.nlm.nih.gov/articles/PMC10670895/)).

## How to Run

### Dependencies

Install in a Python 3.10+ environment (notebook was run on Colab with scikit-learn 1.6):

```bash
pip install gdown numpy pandas matplotlib seaborn scikit-learn itables jupyter
```

The notebook also runs `!pip install itables` inline on first execution.

### Steps

1. Open `Psychometric_Predictor.ipynb` in Jupyter Lab, Jupyter Notebook, or Google Colab.
2. Run cells top to bottom. The first run downloads `dataset.csv` (~2 MB) from Google Drive into the working directory.
3. Intermediate outputs written by the notebook:
   - `mh_T1.csv`
   - `mh_T2.csv`
4. Review printed metrics, heatmaps, and permutation-importance plots in the notebook outputs.

> **Note:** Paths assume the notebook’s working directory is the project folder. On Colab, the default download path is `/content/dataset.csv`.

## Project Structure

```
Downloads/
├── Psychometric_Predictor.ipynb   # Full analysis notebook (sole source artifact)
├── README.md                      # This file
├── Analysis.html                  # Separate HTML export (not generated by the notebook workflow)
└── (generated at runtime)
    ├── adult mental health in indian metro cities 2001-2021(cleaned).csv
    ├── mh_T1.csv
    └── mh_T2.csv
```

There is no packaged Python module, CLI, or trained-model artifact checked into the repo.

---

*Generated from `Psychometric_Predictor.ipynb`.*
