import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
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
# BAYESIAN SEARCH FOR HYPERPARAMETER OPTIMIZATION
#------------------------------------------------------------------

bs_params = {
    'n_estimators': Integer(200, 300), 
    'max_depth': Integer(15, 30), 
    'min_samples_split': Integer(3, 7), 
    'min_samples_leaf': Integer(3, 7), 
    'ccp_alpha': Real(0.005, 0.015), 
    'criterion': Categorical(['entropy']), 
    'max_features': Categorical(['sqrt', 0.1, 0.2, 0.3, 0.4]), 
    'min_impurity_decrease': Real(0.0, 0.006) 
}

bayes_search = BayesSearchCV(rf, bs_params, cv=5, scoring='f1', n_iter=32, verbose=2, random_state=42, n_jobs=-1)
bayes_search.fit(X_train, y_train)
bs_model = bayes_search.best_estimator_
print(f"Best score: {bayes_search.best_score_}")
print(f"Best parameters: {bayes_search.best_params_}")

# Download of the best model and results of the Bayesian search
joblib.dump(bs_model, 'bs_model.pkl')
print("Model saved successfully.")

bs_df = pd.DataFrame(bayes_search.cv_results_)
bs_df.to_csv('bs_dataframe.csv', index=False)
print("Dataframe saved successfully.")

# Loading best model and results of the Bayesian search
# bs_df = pd.read_csv('bs_dataframe.csv')
# bs_model = joblib.load('bs_model.pkl')

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(bs_model, X_train, y_train, cv=5, scoring="accuracy")
print(acc_scores)
print(acc_scores.mean())

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

y_train_pred = cross_val_predict(bs_model, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign (Pred)', 'Malignant (Pred)'],
            yticklabels=['Benign (Real)', 'Malignant (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Confusion Matrix of RF')
plt.show()

#------------------------------------------------------------------
# PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"\nPrecision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")
