import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import joblib
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import os

#------------------------------------------------------------------
# DOWNLOAD & PREPARATION OF THE DATASET
#------------------------------------------------------------------
# Number of classes: 2 (Malignant (M), Benign (B))
# Data per class: 212(M), 357(B)
# Total: 569
# Dimensionality: 30
# Data type: Real and Positive
#------------------------------------------------------------------

# Loading the breast cancer wisconsin dataset from sklearn
dataset = load_breast_cancer()
print(dataset.keys()) 

# Data / Target Separation
X, y = dataset["data"], dataset["target"]
print(X.shape)
print(y.shape)
print(f"\nType of X: {type(X)}")
print(f"Type of y: {type(y)}\n")

# Patient 0 data and diagnosis
print(f"Patient 0 data: {X[0]}")
print(f"Patient 0 diagnosis: {y[0]}\n")

#------------------------------------------------------------------
# DATASET SPLITTING
#------------------------------------------------------------------

# Randomisation & splitting of the dataset into training and test (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}\n")

#------------------------------------------------------------------
# MODEL & DATAFRAME LOADING
#------------------------------------------------------------------

# Uncomment to use rs_model (Random Search)
df = pd.read_csv('dataframes/rs_dataframe.csv')
model = joblib.load('models/rs_model.pkl')

# Uncomment to use gs_model (Grid Search)
# df = pd.read_csv('dataframes/gs_dataframe.csv')
# model = joblib.load('models/gs_model.pkl')

# Uncomment to use bs_model (Bayesian Search)
# df = pd.read_csv('dataframes/bs_dataframe.csv')
# model = joblib.load('models/bs_model.pkl')

os.makedirs('preloaded_graphs', exist_ok=True)

#------------------------------------------------------------------
# HYPERPARAMETER IMPACT ANALYSIS GRAPHICS
#------------------------------------------------------------------

# Winner extraction for graphics
winner = df[df['rank_test_score'] == 1].iloc[0]

parameters = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
              'param_ccp_alpha', 'param_criterion', 'param_max_features', 'param_min_impurity_decrease']

plt.figure()
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.suptitle("Impacto de Hiperparámetros en el F1-Score Medio", fontsize=18, y=0.95)

for i, param in enumerate(parameters):

    plt.subplot(4, 2, i + 1)
    # Sort by numeric values of the parameter for a cleaner line plot
    numeric_col = pd.to_numeric(df[param], errors='coerce')
    if numeric_col.notna().any():
        df_plot = df.assign(_num=numeric_col).sort_values(by="_num").drop(columns="_num")
    else:
        df_plot = df.copy()

    # Transform to string for better x-axis labels & 'None' handling
    df_plot[param] = df[param].astype(object)
    df_plot[param] = df_plot[param].where(df_plot[param].notna(), 'None')
    df_plot[param] = df_plot[param].astype(str)

    # Mean trend line (Blue line)
    sns.lineplot(
    data=df_plot,
    x=param,
    y='mean_test_score',
    errorbar=None,
    color='#2c7bb6',
    linewidth=2,
    marker='o',
    label='Mean Trend'
)
    # Combinations (Red points)
    sns.scatterplot(
    data=df_plot,
    x=param,
    y='mean_test_score',
    alpha=0.3,
    color='#d7191c',
    s=30,
    label='Combinations'
)
    # Winner (Green point) & 'None' handling
    win_param = winner[param] if pd.notna(winner[param]) else 'None'
    win_param = str(win_param)
    win_score = winner['mean_test_score']
    plt.scatter(
        win_param,
        win_score,
        color='#1a9850',
        s=80,
        edgecolors='white',
        zorder=5,
        label='Best Model'
    )

    plt.title(f'F1-Score vs {param.replace("param_", "")}', fontsize=10, fontweight='bold')
    plt.xlabel("")
    plt.ylabel('Mean F1', fontsize=10)
    plt.legend(fontsize=6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('preloaded_graphs/parameter_importance.png')
plt.show()

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (Pred)', 'Malignant (Pred)'],
            yticklabels=['Benign (Real)', 'Malignant (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Cross-Validation Confusion Matrix')
plt.tight_layout()
plt.savefig('preloaded_graphs/confusion_matrix.png')
plt.show()

#------------------------------------------------------------------
# PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"\nPrecision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")
