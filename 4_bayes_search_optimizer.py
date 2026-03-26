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
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret
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

space = {
    'n_estimators': Integer(200, 300), 
    'max_depth': Integer(15, 30), 
    'min_samples_split': Integer(3, 7), 
    'min_samples_leaf': Integer(3, 7), 
    'ccp_alpha': Real(0.005, 0.015), 
    'criterion': Categorical(['entropy']), 
    'max_features': Categorical(['sqrt', 0.1, 0.2, 0.3, 0.4]), 
    'min_impurity_decrease': Real(0.0, 0.006) 
}

bayes_search = BayesSearchCV(rf, space, cv=5, scoring='f1', n_iter=32, verbose=2, random_state=42, n_jobs=-1)
bayes_search.fit(X_train, y_train)
bs_model = bayes_search.best_estimator_
print(f"Best score: {bayes_search.best_score_}")
print(f"Best parameters: {bayes_search.best_params_}")

# Download of the best model and DataFrame of the Bayesian search
os.makedirs('models', exist_ok=True)
os.makedirs('dataframes', exist_ok=True)

joblib.dump(bs_model, 'models/bs_model.pkl')
print("Model saved successfully.")

bs_df = pd.DataFrame(bayes_search.cv_results_)
bs_df.to_csv('dataframes/bs_dataframe.csv', index=False)
print("Dataframe saved successfully.")

# Download of the optimizer results for graphics
joblib.dump(bayes_search.optimizer_results_[0], 'dataframes/bs_optimizer_results.pkl')
print("Optimizer results saved successfully.")

# Loading best model and results of the Bayesian search
# bs_df = pd.read_csv('dataframes/bs_dataframe.csv')
# bs_model = joblib.load('models/bs_model.pkl')

os.makedirs('bs_graphs', exist_ok=True)

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(bs_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")

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
plt.tight_layout()
plt.savefig('bs_graphs/bs_confusion_matrix.png')
plt.show()

#------------------------------------------------------------------
# PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"\nPrecision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")

#------------------------------------------------------------------
# SKOPT CONVERGENCE PLOT
#------------------------------------------------------------------

plt.figure(figsize=(18, 12))
plot_convergence(bayes_search.optimizer_results_[0])
plt.title('Bayesian Search Convergence Plot')
plt.xlabel('Number of Iterations')
plt.ylabel('Best F1 Score')
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96)
plt.savefig('bs_graphs/bayes_convergence_plot.png')
plt.show()

#------------------------------------------------------------------
# SKOPT PLOT EVALUATION
#------------------------------------------------------------------

plt.figure(figsize=(18, 12))
plot_evaluations(bayes_search.optimizer_results_[0])
plt.suptitle('Bayesian Search Evaluations Plot', fontsize=16)
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96)
plt.savefig('bs_graphs/bayes_evaluations_plot.png')
plt.show()

#------------------------------------------------------------------
# SKOPT OBJECTIVE PLOT
#------------------------------------------------------------------

plt.figure(figsize=(20, 14))
plot_objective(bayes_search.optimizer_results_[0])
plt.suptitle('Bayesian Search Objective Plot', fontsize=16)
plt.tight_layout(pad=3.5)
plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
plt.savefig('bs_graphs/bayes_objective_plot.png')
plt.show()

#------------------------------------------------------------------
# SKOPT REGRET PLOT
#------------------------------------------------------------------

plt.figure(figsize=(18, 12))
plot_regret(bayes_search.optimizer_results_[0])
plt.title('Bayesian Search Regret Plot')
plt.xlabel('Number of Iterations')
plt.ylabel('Regret')
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96)
plt.savefig('bs_graphs/bayes_regret_plot.png')
plt.show()

