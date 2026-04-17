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
# dataset = load_breast_cancer()
# print(dataset.keys()) 

# # Data / Target Separation
# X, y = dataset["data"], dataset["target"]
# print(X.shape)
# print(y.shape)
# print(f"Type of X: {type(X)}")
# print(f"Type of y: {type(y)}")

# # Patient 0 data and diagnosis
# print(f"Patient 0 data: {X[0]}")
# print(f"Patient 0 diagnosis: {y[0]}")

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
print(f"Shape of y_test: {y_test.shape}")

#------------------------------------------------------------------
# MODEL CREATION 
#------------------------------------------------------------------

# Random Forest Classifier with constant hyperparameters
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', verbose=1, random_state=42)

#------------------------------------------------------------------
# RANDOM SEARCH FOR HYPERPARAMETER OPTIMIZATION 
#------------------------------------------------------------------

rs_params = {
    'n_estimators': [50, 100, 200, 500], # Number of trees
    # Reduction of Overfitting
    'max_depth': [3, 5, 10, 20, None], # Max depth of each tree
    'min_samples_split': [2, 5, 10], # Min number of samples for node splitting
    'min_samples_leaf': [2, 5, 10, 20], # Min number of samples per leaf
    'ccp_alpha': [0.0, 0.001, 0.01, 0.02, 0.05], # Cost Complexity Pruning
    # Quality of split
    'criterion': ['gini', 'entropy'], # Quality of split (Gini vs Entropy)
    'max_features': [None, 'sqrt', 'log2', 0.5], # Number of random features per tree
    'min_impurity_decrease': [0.0, 0.01, 0.03, 0.05], # Min impurity decrease for node splitting
}

random_search = RandomizedSearchCV(rf, rs_params, n_iter=200, cv=5, random_state=42, scoring='recall') # scoring=[f1,recall,precission,accuracy]
random_search.fit(X_train, y_train)
rs_model = random_search.best_estimator_ 
print(f"Best Recall score: {random_search.best_score_}\n")
print(f"Best Parameters: {random_search.best_params_}\n")

# Download of the best model and DataFrame of the random search
os.makedirs('models', exist_ok=True)
os.makedirs('dataframes', exist_ok=True)

joblib.dump(rs_model, 'models/rs_model.pkl')
print("Model saved successfully.\n")

rs_df = pd.DataFrame(random_search.cv_results_)
rs_df.to_csv('dataframes/rs_dataframe.csv', index=False)
print("Dataframe saved successfully.\n")

# Loading best model and results of the random search
# rs_df = pd.read_csv('dataframes/rs_dataframe.csv')
# rs_model = joblib.load('models/rs_model.pkl')

os.makedirs('rs_graphs', exist_ok=True)

#------------------------------------------------------------------
# HYPERPARAMETER IMPACT ANALYSIS GRAPHICS
#------------------------------------------------------------------

# Winner extraction for graphics
winner = rs_df[rs_df['rank_test_score'] == 1].iloc[0]

parameters = ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
              'param_ccp_alpha', 'param_criterion', 'param_max_features', 'param_min_impurity_decrease']

plt.figure()
for i, param in enumerate(parameters):

    plt.subplot(4, 2, i + 1)
    # Sort by numeric values of the parameter for a cleaner line plot
    numeric_col = pd.to_numeric(rs_df[param], errors='coerce')
    if numeric_col.notna().any():
        df_plot = rs_df.assign(_num=numeric_col).sort_values(by="_num").drop(columns="_num")
    else:
        df_plot = rs_df.copy()

    # Transform to string for better x-axis labels & 'None' handling
    df_plot[param] = rs_df[param].astype(object)
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
    plt.xlabel("")
    plt.ylabel('Mean Recall', fontsize=10)
    plt.legend(fontsize=6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('rs_graphs/rs_parameter_importance.png')
plt.show()

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(rs_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

y_train_pred = cross_val_predict(rs_model, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (Pred)', 'Malignant (Pred)'],
            yticklabels=['Benign (Real)', 'Malignant (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Cross-Validation Confusion Matrix')
plt.tight_layout()
plt.savefig('rs_graphs/rs_confusion_matrix.png')
plt.show()

#------------------------------------------------------------------
# PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"\nPrecision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")