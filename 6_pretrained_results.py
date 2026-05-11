import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# PREPARATION OF THE DATASET
#------------------------------------------------------------------
# Number of classes: 2 (Urgent (M), Non-Urgent (B))
# Data per class: *************
# Total: *************
# Dimensionality: 88
# Data type: Real and Positive
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

#------------------------------------------------------------------
# HYPERPARAMETER IMPACT ANALYSIS GRAPHICS
#------------------------------------------------------------------

os.makedirs('graphs/pretrained_graphs', exist_ok=True)

# Winner extraction for graphics
winner = df[df['rank_test_score'] == 1].iloc[0]

parameters = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
              'param_ccp_alpha', 'param_criterion', 'param_max_features', 'param_min_impurity_decrease']

for i, param in enumerate(parameters):
    plt.figure()
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

    plt.title(f'Recall vs {param.replace("param_", "")}', fontsize=10, fontweight='bold')
    plt.xlabel(f'{param.replace("param_", "")} values', fontsize=10)
    plt.ylabel('Mean Recall', fontsize=10)
    plt.legend(fontsize=6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'graphs/pretrained_graphs/importance_{param.replace("param_", "")}.png')
    plt.show()
    plt.close()


#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

y_train_pred = cross_val_predict(model, X_train, y_train, cv=10)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Control (Pred)', 'Case (Pred)'],
            yticklabels=['Control (Real)', 'Case (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Cross-Validation Confusion Matrix')
plt.tight_layout()
plt.savefig('graphs/pretrained_graphs/confusion_matrix.png')
plt.show()

#------------------------------------------------------------------
# ACCURACY, PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")
print(f"Precision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")

#------------------------------------------------------------------
# BEST HYPERPARAMETER COMBINATION
#------------------------------------------------------------------

print("\nBest Hyperparameter Combination:")
for param in parameters:
    value = winner[param] if pd.notna(winner[param]) else 'None'
    print(f"{param.replace('param_', '')}: {value}")

#------------------------------------------------------------------
# TIME ANALYSIS
#------------------------------------------------------------------

print(f"\nBest Model Training Time: {winner['mean_fit_time']} seconds")
print(f"Best Model Prediction Time: {winner['mean_score_time']} seconds")
print(f"Total simulation time (all combinations & k-folds): {(df['mean_fit_time'].sum() + df['mean_score_time'].sum())*10} seconds")