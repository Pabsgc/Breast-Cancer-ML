# 🔬 Machine Learning Pipeline for Breast Cancer Detection

A streamlined machine learning pipeline for breast cancer diagnosis using a Random Forest classifier with multiple hyperparameter optimization strategies: Random Search, Grid Search, and Bayesian Optimization.

## 📁 Project Structure

### 🔍 **1_rf_model_creator_random_search.py**
Performs initial model training using **RandomizedSearchCV** for broad hyperparameter exploration. Includes 5-fold cross-validation and evaluates performance with accuracy, precision, recall, F1-score, and confusion matrix.
- **Outputs**:
  - `rs_model.pkl`
  - `rs_dataframe.csv`
  - `📁 rs_graphs/`
    - `rs_parameter_importance.png` 
    - `rs_confusion_matrix.png` 

### 📊 **2_results_load_only.py**
Loads previously trained models and generates comparative visualizations, including confusion matrices and parameter importance plots. Its purpose is purely for convenience.
- **Outputs**:
  - `📁 preloaded_graphs/`

### 🎯 **3_grid_search_optimizer.py**
Uses GridSearchCV for exhaustive hyperparameter tuning with systematic evaluation across the parameter grid.
- **Outputs**:
  - `gs_model.pkl`
  - `gs_dataframe.csv`
  - `📁 gs_graphs/`
    - `gs_confusion_matrix.png` 

### 🧠 **4_bayes_search_optimizer.py**
Applies Bayesian optimization (scikit-optimize) for efficient hyperparameter tuning using a Gaussian Process. The workflow transitions from random exploration to recall-focused optimization.
- **Outputs**:
  - `bs_model.pkl`
  - `bs_dataframe.csv`
  - `📁 bs_graphs/`
    - `bs_confusion_matrix.png` 
    - `bayes_convergence_plot.png` 
    - `bayes_evaluations_plot.png` 
    - `bayes_objective_plot.png` 
    - `bayes_regret_plot.png` 

### 📈 **5_graphics.py**
Generates advanced visual analysis for model interpretability, including feature importance, correlations, and SHAP-based explanations.
- **Outputs**: 
  - `📁 analysis_graphs/`
    - `feature_importance.png` 
    - `permutation_importance.png` 
    - `correlation_heatmap.png` 
    - `pairplot.png` 
    - `beeswarm_plot.png` 
    - `force_plot.html` 
    - `dependence_plot.png` 
    - `waterfall.png` 

## 🗂️ Outputs Overview

| File/Directory | Description |
|----------------|-------------|
| `rs_model.pkl` | Random Search model |
| `gs_model.pkl` | Grid Search model |
| `bs_model.pkl` | Bayesian Search model |
| `rs_dataframe.csv` | Random Search dataframe |
| `gs_dataframe.csv` | Grid Search dataframe |
| `bs_dataframe.csv` | Bayesian Search dataframe |
| `📁 rs_graphs/` | Random Search visualizations |
| `📁 gs_graphs/` | Grid Search visualizations |
| `📁 bs_graphs/` | Bayesian Search visualizations |
| `📁 preloaded_graphs/` | Same as above, without the need to re-train |
| `📁 analysis_graphs/` | Final interpretability & insights |

## 🚀 Usage Workflow

1. Run **Random Search** → `1_rf_model_creator_random_search.py` 
2. Run **Grid Search** → `3_grid_search_optimizer.py`
(Very slow! Bayesian Search is supposed to substitute this method.)
3. Run **Bayesian Optimization** → `4_bayes_search_optimizer.py`
4. Analyze results → `5_graphics.py`

## 🎯 Medical Considerations

- **Recall (Sensitivity)** is prioritized to minimize false negatives
- **F1-Score** is used during optimization for balanced evaluation
- Models should always be validated with clinical expertise and real-world data

---

*Built with scikit-learn, scikit-optimize, matplotlib, seaborn, and SHAP*