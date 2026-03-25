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
from sklearn.inspection import permutation_importance
import shap

#------------------------------------------------------------------
# MODEL & DATAFRAME LOADING
#------------------------------------------------------------------

# Uncomment to use rs_model (Random Search)
df = pd.read_csv('rs_dataframe.csv')
model = joblib.load('rs_model.pkl')

# Uncomment to use gs_model (Grid Search)
# df = pd.read_csv('gs_dataframe.csv')
# model = joblib.load('gs_model.pkl')

# Uncomment to use bs_model (Bayesian Search)
# df = pd.read_csv('bs_dataframe.csv')
# model = joblib.load('bs_model.pkl')

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
# FEATURE IMPORTANCE GRAPH
#------------------------------------------------------------------

feature_names = dataset.feature_names
importances = model.feature_importances_

# DataFrame for bar plot
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    palette='magma',
    hue='Feature',
    legend=False
)
plt.title('Feature Importance', fontsize=15)
plt.xlabel('Importance (Entropy)')
plt.ylabel('Clinical Features')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Adds importance values at the end of each bar
for index, value in enumerate(feature_importance_df['Importance']):
    plt.text(value, index, f' {value:.3f}', va='center')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------
# PERMUTATION IMPORTANCE GRAPH
#------------------------------------------------------------------

# 1. Calculate permutation importance.
result = permutation_importance(
    model, X_test, y_test, # Best model from Grid Search and validation set
    n_repeats=10, # Returns the average of 10 shuffles
    random_state=42,
    n_jobs=-1
)

# 2. Organize the results
feature_names = dataset.feature_names
sorted_idx = result.importances_mean.argsort()

perm_df = pd.DataFrame({
    'Variable': [feature_names[i] for i in sorted_idx],
    'Mean_Importance': result.importances_mean[sorted_idx], # Average value of F1 drop (performance)
    'Std_Deviation': result.importances_std[sorted_idx]
})

# 3. Plot (Horizontal so medical feature names are readable)
plt.figure(figsize=(12, 10))
plt.barh(perm_df['Variable'], perm_df['Mean_Importance'],
         xerr=perm_df['Std_Deviation'], color='mediumseagreen', edgecolor='xkcd:darkgreen', ecolor='xkcd:darkgreen', capsize=5)

plt.title("Final Validation: Permutation Importance (On X_test)", fontsize=15)
plt.xlabel("Drop in F1-Score when shuffling the variable")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

#------------------------------------------------------------------
# CORRELATION HEATMAP GRAPH
#------------------------------------------------------------------

# Select the top 10 features from the model to avoid cluttering the map
top_features = feature_importance_df['Feature'].head(10).tolist()
corr_matrix = pd.DataFrame(X_train, columns=dataset.feature_names)[top_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='inferno', center=0.5, fmt=".2f", linewidths=0.5)
plt.title("Heatmap: Linear Dependency between Top Features", fontsize=15)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------
# PAIRPLOT GRAPH
#------------------------------------------------------------------

# Create a temporary DataFrame with the 4 best features and actual diagnosis
df_temp = pd.DataFrame(X_train, columns=dataset.feature_names)
df_temp['target'] = y_train
best_4 = feature_importance_df['Feature'].head(4).tolist()

sns.pairplot(df_temp, vars=best_4, hue='target', palette='husl', diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle("Visual Interaction: How groups separate according to combinations", y=1.02)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------
# BEESWARM GRAPH
#------------------------------------------------------------------

# 1. Create the explainer
explainer = shap.TreeExplainer(model)

# 2. Calculate SHAP values
# Remove additivity check if it causes decimal precision issues
shap_values = explainer.shap_values(X_test, check_additivity=False)

# 3. THE DIMENSION TRICK:
# Sometimes SHAP returns a list [class0, class1] and sometimes a 3D array
# Let's make sure to extract class 1 (Malignant) correctly
if isinstance(shap_values, list):
    # If it's a list, take element 1
    shap_values_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    # If it's a 3D array (samples, features, classes), take class 1
    shap_values_to_plot = shap_values[:, :, 1]
else:
    # If it's already 2D, leave it as is
    shap_values_to_plot = shap_values

# 4. Plot ensuring X_test matches
# If X_test is a DataFrame, SHAP extracts the names automatically.
# If it's a Numpy Array, use feature_names=data.feature_names
shap.summary_plot(shap_values_to_plot, X_test, feature_names=dataset.feature_names)

#------------------------------------------------------------------
# FORCE PLOT GRAPH
#------------------------------------------------------------------

# 1. Choose the patient (index 0)
i = 13

# 2. Extract the BASE VALUE (E[f(x)])
# If explainer.expected_value is a list, take index 1 (Malignant)
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_val = explainer.expected_value[1]
else:
    base_val = explainer.expected_value

# 3. Extract SHAP VALUES of patient i for class 1
# Use the same logic that worked in summary_plot
if isinstance(shap_values, list):
    # If it's a list [class0, class1], take patient i from class 1
    patient_shap = shap_values[1][i]
elif len(shap_values.shape) == 3:
    # If it's a 3D array [samples, features, classes]
    patient_shap = shap_values[i, :, 1]
else:
    # If it's 2D
    patient_shap = shap_values[i]

# 4. Extract patient DATA (ensuring it's a flat array)
import pandas as pd
if isinstance(X_test, pd.DataFrame):
    patient_data = X_test.iloc[i].values
else:
    patient_data = X_test[i]

# 5. DRAW (Using the low-level function that never fails)
shap.initjs()
shap.force_plot(
    base_val,
    patient_shap,
    patient_data,
    feature_names=list(dataset.feature_names),
    link='logit'
)

#------------------------------------------------------------------
# DEPENDENCE PLOT GRAPH
#------------------------------------------------------------------

# 1. Ensure X_test is a clean DataFrame
X_df = pd.DataFrame(X_test, columns=dataset.feature_names)

# 2. Recalculate SHAP in the most compatible way
# Use TreeExplainer directly on the best model
explainer = shap.TreeExplainer(model)

# Sometimes doesn't work well when loading the model from disk. Use KernelExplainer
# Treats the model as a black box instead of reading the trees internally.
# explainer = shap.KernelExplainer(model.predict_proba, X_test)
shap_values = explainer.shap_values(X_test)

# 3. THE CRITICAL ADJUSTMENT:
# If the error says 'size 2', SHAP has saved the data in a format (classes, samples, features)
# We're going to force the matrix to the correct shape: (samples, features)
if isinstance(shap_values, list):
    # If it's a list of 2 elements, take the one for class 1
    shap_v_final = shap_values[1]
else:
    # If it's a 3D array, extract the dimension for class 1
    shap_v_final = shap_values[:, :, 1]

# 4. The dependency plot WITHOUT RISKS
# Use the numeric index directly on the matrix we just cleaned
idx = list(dataset.feature_names).index("worst perimeter")

shap.dependence_plot(
    idx,
    shap_v_final,
    X_df.values,
    feature_names=dataset.feature_names.tolist()
)

#------------------------------------------------------------------
# WATERFALL PLOT GRAPH
#------------------------------------------------------------------

explainer = shap.TreeExplainer(model)
# 1. KernelExplainer instead of TreeExplainer
# TreeExplainer sometimes fails with models loaded from disk.
# KernelExplainer is slower but INFALLIBLE because it treats the model as a black box.
# Use a sample of X_train as background shap.sample(X_train, 10) (or X_test if small)
#explainer = shap.KernelExplainer(model.predict_proba, X_test)

# 2. Calculate for patient i
i = 13
# Only calculate for ONE patient so it doesn't take forever
shap_values_single = explainer.shap_values(X_test[i:i+1])

# 3. Create the explanation
# Note: In KernelExplainer, the output is [samples, features, classes]
exp = shap.Explanation(
    values=shap_values_single[0][:, 1], # Values for class 1 (Malignant)
    base_values=explainer.expected_value[1],
    data=X_test[i],
    feature_names=dataset.feature_names
)

shap.plots.waterfall(exp)

#------------------------------------------------------------------
# RESULTS & PREDICTIONS
#------------------------------------------------------------------

# Resultados finales de modelo potimizado con X_test e y_test
print(model.score(X_test, y_test))

# Predicciónes por paciente:
print(model.predict_proba(X_test))

# Predicción de paciente concreto:
# i=5
# print("Predicción directa del modelo:", model.predict_proba(X_test[i:i+1]))