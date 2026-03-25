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
print(f"Type of X: {type(X)}")
print(f"Type of y: {type(y)}")

# Patient 0 data and diagnosis
print(f"Patient 0 data: {X[0]}")
print(f"Patient 0 diagnosis: {y[0]}")

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

grid_search = GridSearchCV(rf, gs_params, cv=5, scoring='f1', verbose=2)
grid_search.fit(X_train, y_train)
gs_model = grid_search.best_estimator_
print(f"Best score: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")

# Download of the best model and results of the random search
joblib.dump(gs_model, 'gs_model.pkl')
print("Model saved successfully.")

gs_df = pd.DataFrame(grid_search.cv_results_)
gs_df.to_csv('gs_dataframe.csv', index=False)
print("Dataframe saved successfully.")

# Loading best model and results of the random search
# gs_df = pd.read_csv('gs_dataframe.csv')
# gs_model = joblib.load('gs_model.pkl')

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(gs_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

y_train_pred = cross_val_predict(gs_model, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (Pred)', 'Malignant (Pred)'],
            yticklabels=['Benign (Real)', 'Malignant (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Confusion Matrix of RF')
plt.tight_layout()
plt.show()

#------------------------------------------------------------------
# PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"\nPrecision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")
