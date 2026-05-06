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
print(f"Shape of y_test: {y_test.shape}")

#------------------------------------------------------------------
# MODEL CREATION 
#------------------------------------------------------------------

# Random Forest Classifier with constant hyperparameters
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', verbose=1, random_state=42)

#------------------------------------------------------------------
# GRID SEARCH FOR FINE-TUNING THE BEST MODEL FROM RANDOM SEARCH
#------------------------------------------------------------------

gs_params = {
    'n_estimators': [200, 300], 
    'max_depth': [15, 20, 25, 30], 
    'min_samples_split': [3, 4, 5, 6, 7], 
    'min_samples_leaf': [3, 4, 5, 6, 7], 
    'ccp_alpha': [0.005, 0.008, 0.01, 0.012, 0.015], 
    'criterion': ['entropy'], 
    'max_features': ['sqrt', 0.1, 0.2, 0.3, 0.4], 
    'min_impurity_decrease': [0.0, 0.002, 0.004, 0.006] 
}

grid_search = GridSearchCV(rf, gs_params, cv=10, scoring='recall', verbose=2)
grid_search.fit(X_train, y_train)
gs_model = grid_search.best_estimator_
print(f"Best score: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")

# Download of the best model and DataFrame of the grid search
os.makedirs('models', exist_ok=True)
os.makedirs('dataframes', exist_ok=True)

joblib.dump(gs_model, 'models/gs_model.pkl')
print("Model saved successfully.")

gs_df = pd.DataFrame(grid_search.cv_results_)
gs_df.to_csv('dataframes/gs_dataframe.csv', index=False)
print("Dataframe saved successfully.")

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(gs_model, X_train, y_train, cv=10, scoring="accuracy")
print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

os.makedirs('graphs/gs_graphs', exist_ok=True)

y_train_pred = cross_val_predict(gs_model, X_train, y_train, cv=10)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (Pred)', 'Malignant (Pred)'],
            yticklabels=['Benign (Real)', 'Malignant (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Cross-Validation Confusion Matrix')
plt.tight_layout()
plt.savefig('graphs/gs_graphs/gs_confusion_matrix.png')
plt.show()

#------------------------------------------------------------------
# PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"\nPrecision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")
