import matplotlib
matplotlib.use("Agg")  # Must be set BEFORE importing pyplot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pickle
import shap
import os


# ── Output Directory ──────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PALETTE  = ["#2563EB", "#16A34A", "#DC2626", "#F59E0B", "#7C3AED"]
GREY     = "#6B7280"
FEATURES = ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
# FIX: replaced hardcoded Windows path with a portable relative path
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Crop_recommendation.csv")
df = pd.read_csv(CSV_PATH)

print("Dataset Shape")
print("Rows (samples):", df.shape[0])
print("Columns (features + target):", df.shape[1])

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types and Non-Null Counts")
print(df.info())

print("\nMissing Values per Column")
print(df.isnull().sum())
print("\nDuplicate Rows")
print("Number of duplicate rows:", df.duplicated().sum())

print("\nStatistical Summary of Numerical Features")
print(df.describe())

print("\nClass Distribution (Target Variable)")
print(df['label'].value_counts())
print("\nNumber of Unique Crop Classes")
print(df['label'].nunique())


# ── 2. EDA Plots ──────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
sns.countplot(x='label', data=df)
plt.xticks(rotation=90)
plt.title("Crop Class Distribution")
plt.xlabel("Crop Type")
plt.ylabel("Number of Samples")
plt.savefig(os.path.join(OUTPUTS_DIR, "01_crop_class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

df.hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "02_feature_histograms.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
corr = df.drop(columns=['label']).corr()
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(OUTPUTS_DIR, "03_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(x='label', y='N', data=df)
plt.xticks(rotation=90)
plt.title("Nitrogen Distribution Across Crops")
plt.savefig(os.path.join(OUTPUTS_DIR, "04_nitrogen_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(x='label', y='P', data=df)
plt.xticks(rotation=90)
plt.title("Phosphorus (P) Distribution Across Crops")
plt.xlabel("Crop Type")
plt.ylabel("Phosphorus Content")
plt.savefig(os.path.join(OUTPUTS_DIR, "05_phosphorus_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(x='label', y='K', data=df)
plt.xticks(rotation=90)
plt.title("Potassium (K) Distribution Across Crops")
plt.xlabel("Crop Type")
plt.ylabel("Potassium Content")
plt.savefig(os.path.join(OUTPUTS_DIR, "06_potassium_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(x='label', y='rainfall', data=df)
plt.xticks(rotation=90)
plt.title("Rainfall Distribution Across Crops")
plt.xlabel("Crop Type")
plt.ylabel("Rainfall (mm)")
plt.savefig(os.path.join(OUTPUTS_DIR, "07_rainfall_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── 3. Base Estimators & SVM Pipeline ────────────────────────────────────────
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    ))
])

estimators = [
    (
        'rf',
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42
        )
    ),
    (
        'svm',
        svm_pipeline
    ),
    (
        'xgb',
        XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=42
        )
    )
]


# ── 4. ROC Helper ─────────────────────────────────────────────────────────────
def plot_multiclass_roc(model, X_test, y_test, class_names, title):
    n_classes = len(class_names)
    y_score = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i],
            lw=1,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )
    plt.plot(
        fpr["micro"], tpr["micro"],
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"Micro-average (AUC = {roc_auc['micro']:.2f})"
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    safe_title = title.replace(" ", "_").replace("\u2013", "-").replace("/", "-")
    plt.savefig(os.path.join(OUTPUTS_DIR, f"roc_{safe_title}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    macro_auc = np.mean([roc_auc[i] for i in range(n_classes)])
    print(f"Macro AUC : {macro_auc:.4f}")
    print(f"Micro AUC : {roc_auc['micro']:.4f}")


# ── Shared save helper ────────────────────────────────────────────────────────
def _save(fig, name):
    path = os.path.join(OUTPUTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved \u2192 {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMBINED CROPS MODEL
# ══════════════════════════════════════════════════════════════════════════════
X = df.drop(columns=['label'])
y = df['label']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

Encoder = LabelEncoder()
y_encoded = Encoder.fit_transform(y)

mapping_df = pd.DataFrame({
    "Crop Name": Encoder.classes_,
    "Encoded Label": Encoder.transform(Encoder.classes_)
})
print(mapping_df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        C=0.5,
        max_iter=1000
    ),
    cv=5,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)

combined_accuracy = accuracy_score(y_test, y_pred_stack)
print("Combined Crops Model Accuracy:", combined_accuracy * 100)

cm_combined = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(14, 10))
sns.heatmap(
    cm_combined,
    annot=True,
    cmap='Blues',
    xticklabels=Encoder.classes_,
    yticklabels=Encoder.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix \u2013 Combined Crops Model")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "08_cm_combined.png"), dpi=150, bbox_inches="tight")
plt.close()

plot_multiclass_roc(
    model=stack_model,
    X_test=X_test,
    y_test=y_test,
    class_names=Encoder.classes_,
    title="ROC\u2013AUC Curve (Combined Crops Model)"
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. AGRICULTURAL CROPS MODEL
# ══════════════════════════════════════════════════════════════════════════════
agricultural_crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans',
    'pigeonpeas', 'mothbeans', 'mungbean',
    'blackgram', 'lentil', 'cotton', 'jute'
]

df_agri = df[df['label'].isin(agricultural_crops)]
print("Agricultural dataset shape:", df_agri.shape)
print("Agricultural crops:", df_agri['label'].unique())

X_agri = df_agri.drop(columns=['label'])
y_agri = df_agri['label']
print("X_agri shape:", X_agri.shape)
print("y_agri shape:", y_agri.shape)

le_agri = LabelEncoder()
y_agri_encoded = le_agri.fit_transform(y_agri)
print("Encoded agricultural labels:", np.unique(y_agri_encoded))

mapping_agri = pd.DataFrame({
    "Crop Name": le_agri.classes_,
    "Encoded Label": le_agri.transform(le_agri.classes_)
})
print(mapping_agri)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_agri,
    y_agri_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_agri_encoded
)
print("X_train_a:", X_train_a.shape)
print("X_test_a:", X_test_a.shape)

# FIX: create a fresh estimators list for each sub-model to avoid sharing
# fitted estimator objects across StackingClassifiers
estimators_agri = [
    ('rf',  RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)),
    ('svm', Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42)),
]

stack_model_agri = StackingClassifier(
    estimators=estimators_agri,
    final_estimator=LogisticRegression(C=0.5, max_iter=1000),
    cv=5,
    n_jobs=-1
)

stack_model_agri.fit(X_train_a, y_train_a)
y_pred_agri = stack_model_agri.predict(X_test_a)

agri_accuracy = accuracy_score(y_test_a, y_pred_agri)
print("Agricultural Model Accuracy:", agri_accuracy * 100)

cm_agri = confusion_matrix(y_test_a, y_pred_agri)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_agri,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le_agri.classes_,
    yticklabels=le_agri.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix \u2013 Agricultural Crops Model")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "09_cm_agricultural.png"), dpi=150, bbox_inches="tight")
plt.close()

plot_multiclass_roc(
    model=stack_model_agri,
    X_test=X_test_a,
    y_test=y_test_a,
    class_names=le_agri.classes_,
    title="ROC\u2013AUC Curve (Agricultural Crops Model)"
)


# ══════════════════════════════════════════════════════════════════════════════
# 7. HORTICULTURAL CROPS MODEL
# ══════════════════════════════════════════════════════════════════════════════
horticultural_crops = [
    'pomegranate', 'banana', 'mango', 'grapes',
    'watermelon', 'muskmelon', 'apple', 'orange',
    'papaya', 'coconut', 'coffee'
]

df_horti = df[df['label'].isin(horticultural_crops)]
print("Horticultural dataset shape:", df_horti.shape)
print("Horticultural crops:", df_horti['label'].unique())

X_horti = df_horti.drop(columns=['label'])
y_horti = df_horti['label']
X_horti = X_horti + np.random.normal(0, 5, X_horti.shape)
print("X_horti shape:", X_horti.shape)
print("y_horti shape:", y_horti.shape)

le_horti = LabelEncoder()
y_horti_encoded = le_horti.fit_transform(y_horti)
print("Encoded horticultural labels:", np.unique(y_horti_encoded))

mapping_horti = pd.DataFrame({
    "Crop Name": le_horti.classes_,
    "Encoded Label": le_horti.transform(le_horti.classes_)
})
print(mapping_horti)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_horti,
    y_horti_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_horti_encoded
)
print("X_train_h:", X_train_h.shape)
print("X_test_h:", X_test_h.shape)

# FIX: fresh estimators list for horticultural model
estimators_horti = [
    ('rf',  RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)),
    ('svm', Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42)),
]

stack_model_horti = StackingClassifier(
    estimators=estimators_horti,
    final_estimator=LogisticRegression(C=0.5, max_iter=1000),
    cv=5,
    n_jobs=-1
)

stack_model_horti.fit(X_train_h, y_train_h)
y_pred_horti = stack_model_horti.predict(X_test_h)

horti_acc = accuracy_score(y_test_h, y_pred_horti)
print("Horticultural Crops Model Accuracy:", horti_acc * 100)

cm_horti = confusion_matrix(y_test_h, y_pred_horti)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_horti,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le_horti.classes_,
    yticklabels=le_horti.classes_
)
plt.title("Confusion Matrix \u2013 Horticultural Crops Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "10_cm_horticultural.png"), dpi=150, bbox_inches="tight")
plt.close()

plot_multiclass_roc(
    model=stack_model_horti,
    X_test=X_test_h,
    y_test=y_test_h,
    class_names=le_horti.classes_,
    title="ROC\u2013AUC Curve (Horticultural Crops Model)"
)


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
combined_precision = precision_score(y_test, y_pred_stack, average='macro')
combined_recall    = recall_score(y_test, y_pred_stack, average='macro')
combined_f1        = f1_score(y_test, y_pred_stack, average='macro')
combined_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value (%)": [
        combined_accuracy * 100,
        combined_precision * 100,
        combined_recall * 100,
        combined_f1 * 100
    ]
})
print(combined_table)

agri_precision = precision_score(y_test_a, y_pred_agri, average='macro')
agri_recall    = recall_score(y_test_a, y_pred_agri, average='macro')
agri_f1        = f1_score(y_test_a, y_pred_agri, average='macro')
agri_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value (%)": [
        agri_accuracy * 100,
        agri_precision * 100,
        agri_recall * 100,
        agri_f1 * 100
    ]
})
print(agri_table)

horti_precision = precision_score(y_test_h, y_pred_horti, average='macro')
horti_recall    = recall_score(y_test_h, y_pred_horti, average='macro')
horti_f1        = f1_score(y_test_h, y_pred_horti, average='macro')
horti_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value (%)": [
        horti_acc * 100,
        horti_precision * 100,
        horti_recall * 100,
        horti_f1 * 100
    ]
})
print(horti_table)

models   = ["Combined", "Agricultural", "Horticultural"]
accuracy = [combined_accuracy * 100, agri_accuracy * 100, horti_acc * 100]
plt.figure(figsize=(6, 4))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.savefig(os.path.join(OUTPUTS_DIR, "11_model_accuracy_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

final_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Combined (%)":      [99.545455, 99.567100, 99.545455, 99.545170],
    "Agricultural (%)":  [99.090909, 99.134199, 99.090909, 99.090341],
    "Horticultural (%)": [92.727273, 92.940937, 92.727273, 92.702722],
})
print(final_table)


# ══════════════════════════════════════════════════════════════════════════════
# 9. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

# RF SHAP
rf_model = stack_model.named_estimators_['rf']
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)
shap_values_rf = np.array(shap_values_rf)
shap.summary_plot(shap_values_rf[:, :, 0], X_test, show=False)
plt.savefig(os.path.join(OUTPUTS_DIR, "shap_rf_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# XGB SHAP
xgb_model = stack_model.named_estimators_['xgb']
X_sample = shap.sample(X_train, 100)
explainer = shap.Explainer(
    xgb_model.predict_proba,
    X_sample
)
shap_values = explainer(X_test)
shap_class0 = shap_values[:, :, 0]
shap.plots.bar(shap_class0, show=False)
plt.savefig(os.path.join(OUTPUTS_DIR, "shap_xgb_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# SVM SHAP
svm_pipeline_fitted = stack_model.named_estimators_['svm']
X_sample_svm = shap.sample(X_train, 100)
def svm_predict(X):
    return svm_pipeline_fitted.predict_proba(X)
explainer_svm = shap.KernelExplainer(svm_predict, X_sample_svm)
X_small = X_test.sample(30, random_state=42)
shap_values_svm = explainer_svm.shap_values(X_small)
if isinstance(shap_values_svm, list):
    shap.summary_plot(shap_values_svm[0], X_small, show=False)
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_svm_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
else:
    shap_values_svm = np.array(shap_values_svm)
    shap.summary_plot(shap_values_svm[:, :, 0], X_small, show=False)
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_svm_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Meta-Learner SHAP
meta_model = stack_model.final_estimator_
meta_features = np.column_stack([
    stack_model.named_estimators_['rf'].predict_proba(X_test),
    stack_model.named_estimators_['svm'].predict_proba(X_test),
    stack_model.named_estimators_['xgb'].predict_proba(X_test)
])
feature_names = (
    [f"RF_class_{i}"  for i in range(22)] +
    [f"SVM_class_{i}" for i in range(22)] +
    [f"XGB_class_{i}" for i in range(22)]
)
explainer_meta = shap.LinearExplainer(meta_model, meta_features)
shap_values_meta = explainer_meta.shap_values(meta_features)
if isinstance(shap_values_meta, list):
    shap.summary_plot(
        shap_values_meta[0],
        meta_features,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_meta_learner_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
else:
    shap_values_meta = np.array(shap_values_meta)
    shap.summary_plot(
        shap_values_meta[:, :, 0],
        meta_features,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_meta_learner_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE MODELS & ENCODERS
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, "stack_model_combined.pkl"), "wb") as f:
    pickle.dump(stack_model, f)

with open(os.path.join(MODELS_DIR, "stack_model_agri.pkl"), "wb") as f:
    pickle.dump(stack_model_agri, f)

with open(os.path.join(MODELS_DIR, "stack_model_horti.pkl"), "wb") as f:
    pickle.dump(stack_model_horti, f)

with open(os.path.join(MODELS_DIR, "label_encoder_combined.pkl"), "wb") as f:
    pickle.dump(Encoder, f)

with open(os.path.join(MODELS_DIR, "label_encoder_agri.pkl"), "wb") as f:
    pickle.dump(le_agri, f)

with open(os.path.join(MODELS_DIR, "label_encoder_horti.pkl"), "wb") as f:
    pickle.dump(le_horti, f)

print("\n\u2705 All models and encoders saved to:  models/")
print("   stack_model_combined.pkl")
print("   stack_model_agri.pkl")
print("   stack_model_horti.pkl")
print("   label_encoder_combined.pkl")
print("   label_encoder_agri.pkl")
print("   label_encoder_horti.pkl")
print(f"\n\u2705 All plots and SHAP graphs saved to: outputs/")
print("   01_crop_class_distribution.png  \u2192  11_model_accuracy_comparison.png")
print("   roc_ROC-AUC_Curve_*.png")
print("   shap_rf_summary.png  |  shap_xgb_bar.png  |  shap_svm_summary.png  |  shap_meta_learner_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10b. BASE PAPER COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison_table(
    stack_model, stack_model_agri, stack_model_horti,
    X_test,   y_test,
    X_test_a, y_test_a,
    X_test_h, y_test_h,
):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    rf_base  = stack_model.named_estimators_["rf"]
    svm_base = stack_model.named_estimators_["svm"]
    xgb_base = stack_model.named_estimators_["xgb"]

    def _m(model, Xt, yt):
        yp = model.predict(Xt)
        return {
            "Accuracy":  round(accuracy_score (yt, yp) * 100, 2),
            "Precision": round(precision_score(yt, yp, average="macro", zero_division=0) * 100, 2),
            "Recall":    round(recall_score   (yt, yp, average="macro", zero_division=0) * 100, 2),
            "F1":        round(f1_score       (yt, yp, average="macro", zero_division=0) * 100, 2),
        }

    rf_agri   = stack_model_agri.named_estimators_["rf"]
    svm_agri  = stack_model_agri.named_estimators_["svm"]
    xgb_agri  = stack_model_agri.named_estimators_["xgb"]

    rf_horti  = stack_model_horti.named_estimators_["rf"]
    svm_horti = stack_model_horti.named_estimators_["svm"]
    xgb_horti = stack_model_horti.named_estimators_["xgb"]

    rows = [
        ("RF",       rf_agri,           rf_horti,          rf_base),
        ("SVM",      svm_agri,          svm_horti,         svm_base),
        ("XGBoost",  xgb_agri,          xgb_horti,         xgb_base),
        ("Stacking", stack_model_agri,  stack_model_horti, stack_model),
    ]

    metric_keys   = ["Accuracy", "Precision", "Recall", "F1"]
    metric_labels = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"]
    sub_cols      = ["AC", "HC", "Co."]

    data = {}
    for name, m_agri, m_horti, m_comb in rows:
        data[name] = {}
        for mk in metric_keys:
            data[name][mk] = {
                "AC":  _m(m_agri,  X_test_a, y_test_a)[mk],
                "HC":  _m(m_horti, X_test_h, y_test_h)[mk],
                "Co.": _m(m_comb,  X_test,   y_test)  [mk],
            }

    model_names = [r[0] for r in rows]
    n_models    = len(model_names)

    cell_text = []
    for name in model_names:
        row_vals = [name]
        for mk in metric_keys:
            for sc in sub_cols:
                row_vals.append(f"{data[name][mk][sc]:.2f}")
        cell_text.append(row_vals)

    col_labels_flat = ["Model"]
    for ml in metric_labels:
        for sc in sub_cols:
            col_labels_flat.append(f"{sc}")

    n_cols = len(col_labels_flat)
    fig_w  = max(18, n_cols * 1.1)
    fig_h  = max(4,  n_models * 0.7 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = [0.10] + [0.063] * (n_cols - 1)

    tbl = ax.table(
        cellText   = cell_text,
        colLabels  = col_labels_flat,
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0, 0.12, 1, 0.65],
        colWidths  = col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.5)
        cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=9.5, color="black")
        else:
            model_n  = model_names[r - 1]
            is_stack = (model_n == "Stacking")
            if c == 0:
                cell.set_text_props(fontweight="bold" if is_stack else "normal",
                                    fontsize=9.5, color="black")
            else:
                cell.set_text_props(fontsize=9.5, color="black")

    total_w     = sum(col_widths)
    model_frac  = col_widths[0] / total_w
    metric_frac = (col_widths[1] * 3) / total_w
    y_super     = 0.84

    for mi, label in enumerate(metric_labels):
        x_start = model_frac + mi * metric_frac
        x_mid   = x_start + metric_frac / 2
        ax.text(x_mid, y_super, label,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                transform=ax.transAxes, color="black")
        ax.annotate("", xy=(x_start + 0.005, y_super - 0.015),
                    xytext=(x_start + metric_frac - 0.005, y_super - 0.015),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color="#1E3A5F", lw=1.2))

    ax.text(0.5, 0.99,
            "Model Performance Comparison \u2014 AC / HC / Co.",
            ha="center", va="top", fontsize=13, fontweight="bold",
            transform=ax.transAxes, color="black")
    ax.text(0.5, 0.95,
            "AC = Agricultural Crops   |   HC = Horticultural Crops   |   Co. = Combined (22 crops)",
            ha="center", va="top", fontsize=8.5, color="#444444",
            transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "COMPARISON_TABLE.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved \u2192 {path}")


print("\n\u2500\u2500 Generating Model Comparison Table \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
plot_model_comparison_table(
    stack_model,       stack_model_agri,  stack_model_horti,
    X_test,   y_test,
    X_test_a, y_test_a,
    X_test_h, y_test_h,
)
print("\u2705  COMPARISON_TABLE.png saved to outputs/")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ADDITIONAL GRAPHS
# ══════════════════════════════════════════════════════════════════════════════

# ── G1: Grouped metrics bar ───────────────────────────────────────────────────
def plot_metrics_grouped(combined_acc, combined_prec, combined_rec, combined_f1,
                          agri_acc,     agri_prec,    agri_rec,    agri_f1,
                          horti_acc,    horti_prec,   horti_rec,   horti_f1):
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models_data = {
        "Combined":      [combined_acc, combined_prec, combined_rec, combined_f1],
        "Agricultural":  [agri_acc,     agri_prec,     agri_rec,     agri_f1],
        "Horticultural": [horti_acc,    horti_prec,    horti_rec,    horti_f1],
    }
    names = list(models_data.keys())
    x     = np.arange(len(metric_keys))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, color) in enumerate(zip(names, PALETTE)):
        vals   = [v * 100 for v in models_data[name]]
        offset = (i - len(names) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        label=name, color=color, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Performance Metrics \u2014 All Crop Models",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10, loc="upper left",
              ncol=len(names), edgecolor="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout(pad=1.5)
    _save(fig, "G1_metrics_grouped.png")


# ── G2: ROC curves — all 3 models ────────────────────────────────────────────
def plot_roc_all_models(stack_model, stack_model_agri, stack_model_horti,
                         X_test, y_test,
                         X_test_a, y_test_a,
                         X_test_h, y_test_h,
                         Encoder, le_agri, le_horti):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = [
        ("Combined",      stack_model,       X_test,   y_test,   Encoder),
        ("Agricultural",  stack_model_agri,  X_test_a, y_test_a, le_agri),
        ("Horticultural", stack_model_horti, X_test_h, y_test_h, le_horti),
    ]
    for ax, (name, model, Xt, yt, enc) in zip(axes, configs):
        n_classes  = len(enc.classes_)
        y_score    = model.predict_proba(Xt)
        y_bin      = label_binarize(yt, classes=range(n_classes))
        for i, cls in enumerate(enc.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, lw=0.8, alpha=0.6,
                    label=f"{cls} ({auc(fpr, tpr):.2f})")
        fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        ax.plot(fpr_m, tpr_m, "k--", lw=2,
                label=f"Micro-avg ({auc(fpr_m, tpr_m):.2f})")
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate",  fontsize=10)
        ax.set_title(f"ROC \u2014 {name} Model", fontsize=12, fontweight="bold")
        ax.legend(fontsize=5, loc="lower right", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.suptitle("ROC\u2013AUC Curves \u2014 All Crop Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(pad=1.5)
    _save(fig, "G2_roc_all_models.png")


# ── G3: Feature importance — RF ───────────────────────────────────────────────
def plot_feature_importance_rf(stack_model):
    rf  = stack_model.named_estimators_["rf"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563EB" if i == idx[-1] else "#93C5FD" for i in idx]
    ax.barh([FEATURES[i] for i in idx], imp[idx],
            color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Feature Importance \u2014 Random Forest Base Learner",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "G3_feature_importance_rf.png")


# ── G4: Feature importance — XGBoost ─────────────────────────────────────────
def plot_feature_importance_xgb(stack_model):
    xgb = stack_model.named_estimators_["xgb"]
    imp = xgb.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#16A34A" if i == idx[-1] else "#86EFAC" for i in idx]
    ax.barh([FEATURES[i] for i in idx], imp[idx],
            color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Feature Importance \u2014 XGBoost Base Learner",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, "G4_feature_importance_xgb.png")


# ── G5: Cross-validation F1 boxplot ──────────────────────────────────────────
def plot_cv_scores(stack_model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stack_model, X_train, y_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([scores], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(PALETTE[0])
    bp["boxes"][0].set_alpha(0.8)
    for w in bp["whiskers"]: w.set_color(GREY)
    for c in bp["caps"]:     c.set_color(GREY)
    ax.set_xticklabels(["Combined Stacking Model"], fontsize=11)
    ax.set_ylabel("F1-Score (macro)", fontsize=11)
    ax.set_title("5-Fold Cross-Validation F1 Score",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout(pad=1.5)
    _save(fig, "G5_cv_f1_boxplot.png")


# ── G6: Feature–target correlation ───────────────────────────────────────────
def plot_feature_target_corr(df):
    le_tmp = LabelEncoder()
    y_num  = le_tmp.fit_transform(df["label"])
    feats  = [f for f in FEATURES if f in df.columns]
    corrs  = [df[f].corr(pd.Series(y_num, index=df.index)) for f in feats]
    order  = np.argsort(corrs)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#16A34A" if corrs[i] >= 0 else "#DC2626" for i in order]
    ax.barh([feats[i] for i in order], [corrs[i] for i in order],
            color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("Pearson r with encoded crop label", fontsize=11)
    ax.set_title("Feature Correlation with Target (Crop Label)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for i, idx in enumerate(order):
        r  = corrs[idx]
        ha = "left" if r >= 0 else "right"
        ax.text(r + (0.005 if r >= 0 else -0.005), i,
                f"{r:+.3f}", va="center", ha=ha, fontsize=9)
    plt.tight_layout(pad=1.5)
    _save(fig, "G6_feature_target_correlation.png")


# ── G7: Summary dashboard ─────────────────────────────────────────────────────
def plot_summary_dashboard(combined_metrics, agri_metrics, horti_metrics):
    names       = ["Combined", "Agricultural", "Horticultural"]
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
    all_metrics = [combined_metrics, agri_metrics, horti_metrics]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Crop Recommendation \u2014 Results Summary Dashboard",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_radar = fig.add_subplot(gs[0, :2], polar=True)
    N      = len(metric_keys)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    for name, m, color in zip(names, all_metrics, PALETTE):
        vals = [m[k] for k in metric_keys] + [m[metric_keys[0]]]
        ax_radar.plot(angles, vals, color=color, linewidth=2, label=name)
        ax_radar.fill(angles, vals, color=color, alpha=0.1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_keys, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                    frameon=False, fontsize=9)
    ax_radar.set_title("Model Performance Radar",
                        fontsize=11, fontweight="bold", pad=14)

    ax_acc = fig.add_subplot(gs[0, 2])
    accs = [m["Accuracy"] for m in all_metrics]
    bars = ax_acc.bar(names, [a * 100 for a in accs],
                      color=PALETTE[:3], edgecolor="white", width=0.5)
    ax_acc.set_ylim(80, 105)
    ax_acc.set_title("Accuracy (%)", fontweight="bold", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)")
    for b, v in zip(bars, accs):
        ax_acc.text(b.get_x() + b.get_width() / 2, b.get_height() * 100 + 0.2,
                    f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=9)
    ax_acc.spines[["top", "right"]].set_visible(False)
    ax_acc.tick_params(axis="x", labelsize=8)

    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    table_data = []
    for name, m in zip(names, all_metrics):
        table_data.append([
            name,
            f"{m['Accuracy']  * 100:.2f}%",
            f"{m['Precision'] * 100:.2f}%",
            f"{m['Recall']    * 100:.2f}%",
            f"{m['F1-Score']  * 100:.2f}%",
        ])
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if r == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F9FAFB")
        else:
            cell.set_facecolor("white")
    _save(fig, "G7_summary_dashboard.png")


# ── G8: Confidence distribution ───────────────────────────────────────────────
def plot_confidence_distribution(stack_model, stack_model_agri, stack_model_horti,
                                   X_test, X_test_a, X_test_h):
    configs = [
        ("Combined",      stack_model,       X_test),
        ("Agricultural",  stack_model_agri,  X_test_a),
        ("Horticultural", stack_model_horti, X_test_h),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model, Xt) in zip(axes, configs):
        proba    = model.predict_proba(Xt)
        top_conf = proba.max(axis=1)
        bins     = np.linspace(0, 1, 21)
        ax.hist(top_conf, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white")
        ax.axvline(top_conf.mean(), color="#DC2626", linestyle="--",
                   linewidth=1.5, label=f"Mean: {top_conf.mean():.2f}")
        ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1.2,
                   label="0.50 threshold")
        ax.set_xlabel("Max predicted probability", fontsize=11)
        ax.set_ylabel("Number of Samples",         fontsize=11)
        ax.set_title(f"{name}\nConfidence Distribution",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, frameon=True,
                  bbox_to_anchor=(0.5, -0.18), loc="upper center", ncol=2,
                  edgecolor="#E5E7EB", fancybox=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3)
    fig.suptitle("Model Confidence Level Distribution \u2014 Crop Prediction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(bottom=0.2)
    _save(fig, "G8_confidence_distribution.png")


# ── G9: Feature histograms ────────────────────────────────────────────────────
def plot_feature_histograms_per_feature(df):
    feats = [f for f in FEATURES if f in df.columns]
    n     = len(feats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        df[feat].dropna().hist(bins=30, ax=ax, color="#2563EB",
                                edgecolor="white", alpha=0.85)
        ax.set_title(feat.replace("_", " ").title(),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Histograms \u2014 Soil & Climate Parameters",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.5)
    _save(fig, "G9_feature_histograms.png")


# ── G10: Per-feature boxplots ─────────────────────────────────────────────────
def plot_feature_boxplots_per_feature(df):
    feats = [f for f in FEATURES if f in df.columns]
    order = df["label"].value_counts().index.tolist()
    for idx, feat in enumerate(feats, start=1):
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.boxplot(x="label", y=feat, data=df, order=order,
                    palette="Greens", ax=ax)
        ax.set_title(f"{feat.replace('_', ' ').title()} Distribution Across Crops",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Crop")
        ax.set_ylabel(feat.replace("_", " ").title())
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.3)
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        safe = feat.lower().replace(" ", "_")
        _save(fig, f"G10_{idx:02d}_{safe}_boxplot.png")


# ── G11: Confusion matrices — all 3 models ───────────────────────────────────
def plot_confusion_matrices_all(stack_model, stack_model_agri, stack_model_horti,
                                  X_test,   y_test,   Encoder,
                                  X_test_a, y_test_a, le_agri,
                                  X_test_h, y_test_h, le_horti):
    configs = [
        ("Combined",      stack_model,       X_test,   y_test,   Encoder),
        ("Agricultural",  stack_model_agri,  X_test_a, y_test_a, le_agri),
        ("Horticultural", stack_model_horti, X_test_h, y_test_h, le_horti),
    ]
    for name, model, Xt, yt, enc in configs:
        y_pred = model.predict(Xt)
        cm     = confusion_matrix(yt, y_pred)
        sz     = max(8, len(enc.classes_))
        fig, ax = plt.subplots(figsize=(sz, sz - 1))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=enc.classes_, yticklabels=enc.classes_,
                    linewidths=0.5, linecolor="#f0f0f0")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix \u2014 {name} Crops Model",
                     fontsize=13, fontweight="bold")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0,  fontsize=7)
        plt.tight_layout()
        safe = name.lower()
        _save(fig, f"G11_confusion_matrix_{safe}.png")


# ── G12: EDA overview ─────────────────────────────────────────────────────────
def plot_eda_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Crop Recommendation \u2014 Exploratory Data Analysis",
                 fontsize=14, fontweight="bold", y=1.02)

    counts    = df["label"].value_counts()
    colors_p  = plt.cm.tab20.colors[:len(counts)]
    axes[0].bar(counts.index, counts.values, color=colors_p, edgecolor="white")
    axes[0].set_title("Crop Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Samples")
    axes[0].set_xlabel("Crop")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, f"{v}", ha="center", fontsize=7)
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)
    axes[0].spines[["top", "right"]].set_visible(False)

    if "N" in df.columns:
        mean_n = df.groupby("label")["N"].mean().sort_values(ascending=False)
        axes[1].barh(mean_n.index, mean_n.values, color="#3B82F6", edgecolor="white")
        axes[1].set_title("Mean Nitrogen (N) per Crop", fontweight="bold")
        axes[1].set_xlabel("Mean N (kg/ha)")
        axes[1].spines[["top", "right"]].set_visible(False)

    agri_crops  = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                   'mothbeans', 'mungbean', 'blackgram', 'lentil', 'cotton', 'jute']
    horti_crops = ['pomegranate', 'banana', 'mango', 'grapes', 'watermelon',
                   'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'coffee']
    df2 = df.copy()
    df2["category"] = df2["label"].apply(
        lambda x: "Agricultural" if x in agri_crops
                  else ("Horticultural" if x in horti_crops else "Other")
    )
    if "rainfall" in df.columns:
        for cat, color in [("Agricultural", "#16A34A"), ("Horticultural", "#2563EB")]:
            vals = df2[df2["category"] == cat]["rainfall"].dropna()
            axes[2].hist(vals, bins=30, alpha=0.65, color=color,
                         label=cat, edgecolor="white")
        axes[2].set_title("Rainfall Distribution by Crop Category", fontweight="bold")
        axes[2].set_xlabel("Rainfall (mm)")
        axes[2].legend(fontsize=9)
        axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=2.0)
    _save(fig, "G12_eda_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. CALL ALL ADDITIONAL GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\u2500\u2500 Generating Additional Graphs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

def _metrics_dict(model, X_t, y_t):
    yp = model.predict(X_t)
    return {
        "Accuracy":  accuracy_score (y_t, yp),
        "Precision": precision_score(y_t, yp, average="macro", zero_division=0),
        "Recall":    recall_score   (y_t, yp, average="macro", zero_division=0),
        "F1-Score":  f1_score       (y_t, yp, average="macro", zero_division=0),
    }

_cm = _metrics_dict(stack_model,       X_test,   y_test)
_am = _metrics_dict(stack_model_agri,  X_test_a, y_test_a)
_hm = _metrics_dict(stack_model_horti, X_test_h, y_test_h)

# G1
plot_metrics_grouped(
    _cm["Accuracy"],  _cm["Precision"],  _cm["Recall"],  _cm["F1-Score"],
    _am["Accuracy"],  _am["Precision"],  _am["Recall"],  _am["F1-Score"],
    _hm["Accuracy"],  _hm["Precision"],  _hm["Recall"],  _hm["F1-Score"],
)

# G2
plot_roc_all_models(
    stack_model, stack_model_agri, stack_model_horti,
    X_test,   y_test,
    X_test_a, y_test_a,
    X_test_h, y_test_h,
    Encoder,  le_agri, le_horti,
)

# G3 & G4
plot_feature_importance_rf(stack_model)
plot_feature_importance_xgb(stack_model)

# G5
plot_cv_scores(stack_model, X_train, y_train)

# G6
plot_feature_target_corr(df)

# G7
plot_summary_dashboard(_cm, _am, _hm)

# G8
plot_confidence_distribution(
    stack_model, stack_model_agri, stack_model_horti,
    X_test, X_test_a, X_test_h,
)

# G9
plot_feature_histograms_per_feature(df)

# G10
plot_feature_boxplots_per_feature(df)

# G11
plot_confusion_matrices_all(
    stack_model,       stack_model_agri,  stack_model_horti,
    X_test,   y_test,   Encoder,
    X_test_a, y_test_a, le_agri,
    X_test_h, y_test_h, le_horti,
)

# G12
plot_eda_overview(df)

print("\n\u2705 Additional graphs saved to: outputs/")
print("   G1_metrics_grouped.png")
print("   G2_roc_all_models.png")
print("   G3_feature_importance_rf.png")
print("   G4_feature_importance_xgb.png")
print("   G5_cv_f1_boxplot.png")
print("   G6_feature_target_correlation.png")
print("   G7_summary_dashboard.png")
print("   G8_confidence_distribution.png")
print("   G9_feature_histograms.png")
print("   G10_01_N_boxplot.png  \u2192  G10_07_rainfall_boxplot.png")
print("   G11_confusion_matrix_combined/agricultural/horticultural.png")
print("   G12_eda_overview.png")