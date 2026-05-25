2. Methodology
2.1 Generation Framework
Probabilistic Sampling with Epidemiological Constraints

Monte Carlo approach extracting parameters from WHO reports, PLOS meta-analyses, and African cohort studies:

For each sample i:
  1. Age_i ~ Categorical(0-14: 15%, 15-24: 10%, 25-34: 30%, 35-49: 20%, 50+: 25%)
  2. Sex_i ~ Bernoulli(0.55) for Male
  3. HIV_i ~ Bernoulli(0.34 if age 15-49, else 0.20)  # SSA co-infection rate
  4. If HIV+: CD4_i ~ Truncated_Normal(350, 200, min=10, max=1200)
  5. TB_prob_i = f(age, HIV, residence, prior_TB, BMI, overcrowding)
  6. TB_i ~ Bernoulli(TB_prob_i)
  7. If TB+:
       - Type_i ~ Bernoulli(0.75) for Pulmonary vs Extrapulmonary
       - If Pulmonary & HIV-: Smear+ ~ Bernoulli(0.65)
       - If Pulmonary & HIV+: Smear+ ~ Bernoulli(0.45)
       - If new_case: MDR ~ Bernoulli(0.02)
       - If retreatment: MDR ~ Bernoulli(0.12)
  8. Outcome_i ~ f(TB, HIV, MDR, age, treatment_adherence)
2.2 Sub-Saharan Africa Parameters
Key differences from global distributions:

Parameter	SSA Value	Global Value	Source
TB-HIV co-infection	34%	8%	WHO African Region, 2004
MDR-TB (new cases)	2.0% (95% CI: 1.7-2.4%)	4.1%	PLOS One meta-analysis, 2017
Female TB cases (high-HIV)	50-55%	35%	NCBI SSA studies
Smear+ in HIV+ patients	45%	65%	African cohorts
Case-fatality (HIV+)	15-25%	5-8%	Malawi, Zimbabwe studies
Additional SSA-specific factors: Overcrowding (60% urban, 40% rural), Previous TB (8% in general population, 15% in HIV+), Low BMI (<18.5: 25% in TB+, 12% in TB-).

2.3 TB Probability Model
Additive risk calculation:

P_base = tb_prevalence  # Typically 0.30 in high-burden setting

# HIV status (6× incidence rate ratio):
if hiv_positive:
    P_base *= 6.0
    if cd4_count < 200:
        P_base *= 1.5  # Advanced immunosuppression

# Socioeconomic factors:
if overcrowding:
    P_base *= 1.4
if low_bmi:
    P_base *= 1.3
if previous_tb:
    P_base *= 2.5  # Reactivation/reinfection risk

# Age modulation (peak 25-34):
if age_category == "25-34":
    P_base *= 1.2
elif age_category in ["0-14", "50+"]:
    P_base *= 0.8

# Urban vs rural:
if residence == "Urban":
    P_base *= 1.1  # Higher transmission

P_final = min(P_base, 0.95)  # Realism ceiling


4. Model Training Protocol
4.1 Recommended Pipeline
Step 1: Data Preparation

import pandas as pd
from sklearn.model_selection import train_test_split

# Load training data
df = pd.read_csv('tb_ssa_large_5000.csv')

# Select features
feature_cols = [
    'age', 'sex', 'residence', 'overcrowding', 'bmi_category',
    'previous_tb_treatment', 'hiv_status', 'cd4_count',  
    'cough_duration', 'hemoptysis', 'night_sweats', 'fever',
    'weight_loss_kg', 'smear_status', 'esr_mm_hr', 'hemoglobin_g_dl'
]

# Encode categoricals
from sklearn.preprocessing import LabelEncoder
le_dict = {}
for col in ['sex', 'residence', 'bmi_category', 'hiv_status', 'smear_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Handle missing CD4 (non-HIV patients)
df['cd4_count'] = df['cd4_count'].fillna(0)

X = df[feature_cols]
y_tb = df['tb_status'].map({'Positive': 1, 'Negative': 0})
y_outcome = df['treatment_outcome']  # For outcome prediction
Step 2: Task Selection

Choose one of three prediction tasks:

TB Detection (TB+ vs TB-)
Co-infection Prediction (TB+HIV+ vs TB+HIV-)
MDR-TB Prediction (MDR vs drug-sensitive)
Treatment Outcome (Cured/Died/Failed/LTFU)
Step 3: Model Training (Example: TB-HIV Co-infection)

# Filter to TB-positive patients only
tb_positive = df[df['tb_status'] == 'Positive'].copy()
X_coinfection = tb_positive[['age', 'sex', 'cd4_count', 'bmi_category', 
                              'overcrowding', 'previous_tb_treatment']]
y_coinfection = tb_positive['hiv_status'].map({'Positive': 1, 'Negative': 0})

# Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_coinfection, y_coinfection, test_size=0.2, 
    stratify=y_coinfection, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score, classification_report
y_prob = model.predict_proba(X_val)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_val, y_prob):.3f}")
print(classification_report(y_val, model.predict(X_val)))
4.2 Hyperparameter Tuning
Grid Search for Random Forest:

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
5. Evaluation Protocol
5.1 Primary Metrics
For TB-HIV Co-infection Prediction:

Metric	Target	Clinical Rationale
AUC-ROC	≥0.82	Overall discriminative ability
Sensitivity	≥75%	Identify co-infected for ART initiation
Specificity	≥70%	Avoid unnecessary HIV testing
NPV	≥85%	Confidence in negative results
For MDR-TB Prediction:

Metric	Target	Clinical Rationale
AUC-ROC	≥0.75	Challenging task (low prevalence)
Sensitivity	≥60%	Catch MDR cases for second-line therapy
PPV	≥15%	Balance against DST resource use
5.2 Final Evaluation Code
# Load test set
test_df = pd.read_csv('tb_ssa_test_2000.csv')
# ... (same preprocessing as training)

# Predict
y_test_prob = final_model.predict_proba(X_test)[:, 1]
y_test_pred = final_model.predict(X_test)

# Comprehensive metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nTest Set Performance:")
print(f"  Sensitivity: {sensitivity:.3f}")
print(f"  Specificity: {specificity:.3f}")
print(f"  PPV: {ppv:.3f}")
print(f"  NPV: {npv:.3f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_test_prob):.3f}")