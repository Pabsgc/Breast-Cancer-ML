import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

#------------------------------------------------------------------
# MODEL & DATAFRAME LOADING
#------------------------------------------------------------------

# Uncomment to use rs_model (Random Search)
# model = joblib.load('models/rs_model.pkl')

# Uncomment to use gs_model (Grid Search)
# model = joblib.load('models/gs_model.pkl')

# Uncomment to use bs_model (Bayesian Search)
model = joblib.load('models/bs_model.pkl')

#------------------------------------------------------------------
# PREPARATION OF THE DATASET
#------------------------------------------------------------------

# Loading extracted features dataset from CSV
dataset = pd.read_csv('dataset/feature_extraction_dataset.csv')
print(dataset.keys())

# Data / Target Separation
# X keeps everything except 'label' and 'filename' columns

y = dataset['label']
X = dataset.drop(columns=['label', 'filename'])

# Dataset overview
print(f"X shape (samples, features): {X.shape}")
print(f"Type of X: {type(X)}")
print(f"y shape (labels): {y.shape}")
print(f"Type of y: {type(y)}")

# Patient 0 data (using .iloc for pandas indexing)
print(f"Patient 0 data: {X.iloc[0]}")
print(f"Patient 0 diagnosis: {y.iloc[0]}")

# Check for NaNs
if X.isnull().values.any():
    print("Warning: NaNs found. Filling with 0.")
    X = X.fillna(0)

#------------------------------------------------------------------
# DATASET SPLITTING
#------------------------------------------------------------------

# Randomisation & splitting of the dataset into training and test sets
# Use stratify=y to preserve class balance in the small dataset.
# El modelo ya está entrenado; este split solo se usa para la curva ROC de visualización.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=42
)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Test label counts: {y_test.value_counts().to_dict()}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auroc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"AUROC CV mean: {auroc_scores.mean():.4f}")
print(f"AUROC CV std: {auroc_scores.std():.4f}")

#------------------------------------------------------------------
# ROC CURVE, AUROC AND YOUDEN INDEX
#------------------------------------------------------------------

# Convert labels to binary 0/1 for ROC calculation
lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test).ravel()

# Calculate probabilities for the positive class
if hasattr(model, 'predict_proba'):
    y_score = model.predict_proba(X_test)[:, 1]
else:
    raise AttributeError('El modelo no tiene método predict_proba. Use un clasificador compatible.')

fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
roc_auc = auc(fpr, tpr)

youden_index = tpr - fpr
best_index = np.argmax(youden_index)
best_threshold = thresholds[best_index]
best_youden = youden_index[best_index]
best_sensitivity = tpr[best_index]
best_specificity = 1 - fpr[best_index]

print(f"AUROC (test set): {roc_auc:.4f}")
print(f"Youden's J: {best_youden:.4f}")
print(f"Mejor umbral (threshold): {best_threshold:.4f}")
print(f"Sensibilidad (TPR) en ese umbral: {best_sensitivity:.4f}")
print(f"Especificidad (1 - FPR) en ese umbral: {best_specificity:.4f}")

if roc_auc == 1.0:
    print("Nota: con un conjunto de prueba muy pequeño y perfectamente ordenado, el AUROC de ese split puede ser 1.0.")
    print("      Usa la media de AUROC CV anterior para una evaluación más representativa.")

#------------------------------------------------------------------
# SAVE ROC CURVE PLOT
#------------------------------------------------------------------

os.makedirs('graphs/analysis_graphs', exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.scatter(
    fpr[best_index],
    tpr[best_index],
    color='red',
    s=80,
    label=f"Youden J = {best_youden:.4f}\nThreshold = {best_threshold:.4f}"
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/analysis_graphs/roc_curve.png')
plt.show()
