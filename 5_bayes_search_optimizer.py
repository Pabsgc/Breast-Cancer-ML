import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', verbose=2, random_state=42)

#------------------------------------------------------------------
# BAYESIAN SEARCH FOR HYPERPARAMETER OPTIMIZATION
#------------------------------------------------------------------

space = {
    'n_estimators': Integer(100, 800),
    'max_depth': Categorical([None] + list(range(10, 40))),
    'min_samples_split': Integer(5, 15),
    'min_samples_leaf': Integer(1, 10),
    'ccp_alpha': Real(0.001, 0.3),
    'criterion': Categorical(['entropy', 'gini']),
    'max_features': Categorical(['sqrt', 'log2', 0.1]),
    'min_impurity_decrease': Real(0.0, 0.005)
}

bayes_search = BayesSearchCV(rf, space, cv=10, scoring='recall', n_iter=50, random_state=42, n_jobs=-1)
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

#------------------------------------------------------------------
# ACCURACY SCORES
#------------------------------------------------------------------

acc_scores = cross_val_score(bs_model, X_train, y_train, cv=10, scoring="accuracy")

#------------------------------------------------------------------
# CONFUSION MATRIX
#------------------------------------------------------------------

os.makedirs('graphs/bs_graphs', exist_ok=True)

y_train_pred = cross_val_predict(bs_model, X_train, y_train, cv=10)
cm = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Control (Pred)', 'Case (Pred)'],
            yticklabels=['Control (Real)', 'Case (Real)'])
plt.xlabel('Prediction')
plt.ylabel('Reality')
plt.title('Cross-Validation Confusion Matrix')
plt.tight_layout()
plt.savefig('graphs/bs_graphs/bs_confusion_matrix.png')
plt.show()
plt.close()

#------------------------------------------------------------------
# ACCURACY, PRECISION, RECALL AND F1-SCORE
#------------------------------------------------------------------

print(f"Accuracy Scores: {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean()}")
print(f"Precision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1-Score: {f1_score(y_train, y_train_pred)}")

#------------------------------------------------------------------
# SKOPT CONVERGENCE PLOT
#------------------------------------------------------------------

plot_convergence(bayes_search.optimizer_results_[0])
plt.title('Bayesian Search Convergence Plot')
plt.xlabel('Number of Iterations')
plt.ylabel('Best Recall Score')
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96)
plt.savefig('graphs/bs_graphs/bayes_convergence_plot.png')
plt.show()
plt.close()

#------------------------------------------------------------------
# SKOPT PLOT EVALUATION
#------------------------------------------------------------------

plot_evaluations(bayes_search.optimizer_results_[0])
plt.suptitle('Bayesian Search Evaluations Plot', fontsize=16)
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96)
plt.savefig('graphs/bs_graphs/bayes_evaluations_plot.png')
plt.show()
plt.close()

#------------------------------------------------------------------
# SKOPT OBJECTIVE PLOT
#------------------------------------------------------------------


plot_objective(bayes_search.optimizer_results_[0])
plt.suptitle('Bayesian Search Objective Plot', fontsize=16)
plt.tight_layout(pad=3.5)
plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
plt.savefig('graphs/bs_graphs/bayes_objective_plot.png')
plt.show()
plt.close()

#------------------------------------------------------------------
# SKOPT REGRET PLOT
#------------------------------------------------------------------

plot_regret(bayes_search.optimizer_results_[0])
plt.title('Bayesian Search Regret Plot')
plt.xlabel('Number of Iterations')
plt.ylabel('Regret')
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.96)
plt.savefig('graphs/bs_graphs/bayes_regret_plot.png')
plt.show()
plt.close()

