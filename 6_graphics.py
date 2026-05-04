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
from sklearn.inspection import permutation_importance
import shap
import os
import re

#------------------------------------------------------------------
# MODEL & DATAFRAME LOADING
#------------------------------------------------------------------

# Uncomment to use rs_model (Random Search)
# df = pd.read_csv('dataframes/rs_dataframe.csv')
# model = joblib.load('models/rs_model.pkl')

# Uncomment to use gs_model (Grid Search)
# df = pd.read_csv('dataframes/gs_dataframe.csv')
# model = joblib.load('models/gs_model.pkl')

# Uncomment to use bs_model (Bayesian Search)
df = pd.read_csv('dataframes/bs_dataframe.csv')
model = joblib.load('models/bs_model.pkl')

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
filenames = dataset['filename']
X = dataset.drop(columns=['label', 'filename'])

# Dataset overview
print(f"X shape (samples, features): {X.shape}")
print(f"Type of X: {type(X)}")
print(f"y shape (labels): {y.shape}")
print(f"Type of y: {type(y)}")

# Patient 0 data (using .iloc for pandas indexing)
print(f"Patient 0 data: {X.iloc[0]}")
print(f"Patient 0 diagnosis: {y.iloc[0]}")
print(f"Patient 0 filename: {filenames.iloc[0]}")

# Check for NaNs
if X.isnull().values.any():
    print("Warning: NaNs found. Filling with 0.")
    X = X.fillna(0)

#------------------------------------------------------------------
# DATASET SPLITTING
#------------------------------------------------------------------

# Randomisation & splitting of the dataset into training and test (10%) sets
X_train, X_test, y_train, y_test, filename_train, filename_test = train_test_split(
    X,
    y,
    filenames,
    test_size=0.1,
    random_state=42
)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of filename_train: {filename_train.shape}")
print(f"Shape of filename_test: {filename_test.shape}")

#------------------------------------------------------------------
# MODEL CREATION 
#------------------------------------------------------------------

# Random Forest Classifier with constant hyperparameters
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', verbose=1, random_state=42)

#------------------------------------------------------------------
# FEATURE IMPORTANCE GRAPH
#------------------------------------------------------------------

os.makedirs('graphs/analysis_graphs', exist_ok=True)

feature_names = X.columns
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
plt.savefig('graphs/analysis_graphs/feature_importance.png')
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
feature_names = X.columns
sorted_idx = result.importances_mean.argsort()

perm_df = pd.DataFrame({
    'Variable': [feature_names[i] for i in sorted_idx],
    'Mean_Importance': result.importances_mean[sorted_idx], # Average value of recall drop (performance)
    'Std_Deviation': result.importances_std[sorted_idx]
})

# 3. Plot (Horizontal so medical feature names are readable)
plt.figure(figsize=(12, 10))
plt.barh(perm_df['Variable'], perm_df['Mean_Importance'],
         xerr=perm_df['Std_Deviation'], color='mediumseagreen', edgecolor='xkcd:darkgreen', ecolor='xkcd:darkgreen', capsize=5)

plt.title("Final Validation: Permutation Importance (On X_test)", fontsize=15)
plt.xlabel("Drop in Recall when shuffling the variable")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('graphs/analysis_graphs/permutation_importance.png')
plt.show()

#------------------------------------------------------------------
# CORRELATION HEATMAP GRAPH
#------------------------------------------------------------------

# Select the top 10 features from the model to avoid cluttering the map
top_features = feature_importance_df['Feature'].head(10).tolist()
corr_matrix = pd.DataFrame(X_train, columns=X.columns)[top_features].corr()

# Get the min and max values for proper color scaling
vmin = corr_matrix.values.min()
vmax = corr_matrix.values.max()
# Center the colormap at 0 (so white is at 0)
abs_max = max(abs(vmin), abs(vmax))

plt.figure(figsize=(12, 10))
# Create a mask for the upper triangle (excluding diagonal)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-abs_max, vmax=abs_max, center=0, fmt=".2f", linewidths=0.5, mask=mask)
plt.title("Heatmap: Linear Dependency between Top Features", fontsize=15)
plt.tight_layout()
plt.savefig('graphs/analysis_graphs/correlation_heatmap.png')
plt.show()

#------------------------------------------------------------------
# PAIRPLOT GRAPH
#------------------------------------------------------------------

# Create a temporary DataFrame with the 4 best features and actual diagnosis
df_temp = pd.DataFrame(X_train, columns=X.columns)
df_temp['target'] = y_train
best_4 = feature_importance_df['Feature'].head(4).tolist()

sns.pairplot(df_temp, vars=best_4, hue='target', palette='husl', diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle("Visual Interaction: How groups separate according to combinations", y=1.02)
plt.tight_layout()
plt.savefig('graphs/analysis_graphs/pairplot.png')
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
shap.summary_plot(shap_values_to_plot, X_test, feature_names=X.columns)
plt.savefig('graphs/analysis_graphs/beeswarm_plot.png')
plt.close()

#------------------------------------------------------------------
# FORCE PLOT GRAPH
#------------------------------------------------------------------

# 1. Choose the patient (index 0)
i = 0

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
force_plot_html = shap.force_plot(
    base_val,
    patient_shap,
    patient_data,
    feature_names=list(X.columns),
    link='logit'
)

# Save HTML manually to avoid Windows file path issues
with open('graphs/analysis_graphs/force_plot.html', 'w', encoding='utf-8') as f:
    f.write(force_plot_html.html())
plt.close()  # Close any figure created by force_plot

#------------------------------------------------------------------
# DEPENDENCE PLOT GRAPH
#------------------------------------------------------------------

# 1. Ensure X_test is a clean DataFrame
X_df = pd.DataFrame(X_test, columns=X.columns)

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
idx = list(X.columns).index("mfcc1_sma3_amean")

shap.dependence_plot(
    idx,
    shap_v_final,
    X_df.values,
    feature_names=X.columns.tolist()
)
plt.savefig('graphs/analysis_graphs/dependence_plot.png')
plt.close()

#------------------------------------------------------------------
# WATERFALL PLOT GRAPH
#------------------------------------------------------------------

# 1. Choose the patient (index 0)
i = 0

# Preserve traceability: get the original filename for this test sample
patient_filename = filename_test.iloc[i]
print(f"Generating waterfall plot for test sample index {i} -> filename: {patient_filename}")

# Use the SHAP values from the DEPENDENCE PLOT section (shap_v_final)
# which is already properly extracted for class 1
# Extract patient 0's SHAP values
patient_shap_values = shap_v_final[i]

# Get the base value from the explainer
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_val = explainer.expected_value[1]
else:
    base_val = explainer.expected_value

# Ensure base_val is a float
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(np.array(base_val).flatten()[0])
else:
    base_val = float(base_val)

# Create Explanation object - use the same format that works with shap.plots.waterfall
exp = shap.Explanation(
    values=patient_shap_values,
    base_values=base_val,
    data=X_test.iloc[i].values,
    feature_names=list(X.columns)
)

# Plot and save using filename metadata for traceability
shap.plots.waterfall(exp, show=False)
plt.title(f"Waterfall: {patient_filename}")
output_filename = re.sub(r'[<>:"/\\|?*]', '_', patient_filename)
output_path = f'graphs/analysis_graphs/waterfall_plot_{output_filename}.png'
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()

#------------------------------------------------------------------
# RESULTS & PREDICTIONS
#------------------------------------------------------------------

# Final results of the optimized model with X_test and y_test
print(model.score(X_test, y_test))

# Predictions by patient:
print(model.predict_proba(X_test))

# Prediction for specific patient:
# i=5
# print("Direct model prediction:", model.predict_proba(X_test[i:i+1]))