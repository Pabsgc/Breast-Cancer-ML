# 🔬 Random Forest Classifier for Breast Cancer Detection

A comprehensive machine learning pipeline for breast cancer diagnosis using Random Forest classification with multiple hyperparameter optimization strategies.

## 📁 Project Structure

### 🔍 **1_rf_model_creator_random_search.py**
Creates Random Forest model using RandomizedSearchCV for initial hyperparameter exploration.
- **Features**: 5-fold cross-validation, confusion matrix, accuracy scores, precision/recall/F1 metrics
- **Results saved in**: `📁 rs_graphs/`
  - `rs_parameter_importance.png` - Hyperparameter impact visualization
  - `rs_confusion_matrix.png` - Model performance matrix

### 📊 **2_results_load_only.py**
Loads pre-trained models and generates analysis graphics for model comparison.
- **Features**: Parameter importance plots, confusion matrices, performance metrics
- **Results saved in**: `📁 analysis_graphs/`
  - `parameter_importance.png` - Hyperparameter analysis
  - `confusion_matrix.png` - Performance visualization

### 🎯 **3_grid_search_optimizer.py**
Utilizes GridSearchCV (exhaustive search) for systematic hyperparameter optimization.
- **Features**: Monte Carlo-style parameter grid exploration, comprehensive evaluation
- **Results saved in**: `📁 gs_graphs/`
  - `gs_confusion_matrix.png` - Grid search performance matrix

### 🧠 **4_bayes_search_optimizer.py**
Implements Bayesian optimization using scikit-optimize for intelligent hyperparameter tuning.
- **Features**: Gaussian Process surrogate modeling, acquisition functions, convergence plots
- **Optimization Strategy**: RandomSearchCV → F1 exploration → BayesSearchCV → Recall optimization
- **Results saved in**: `📁 bs_graphs/`
  - `bayes_convergence_plot.png` - Optimization convergence
  - `bayes_evaluations_plot.png` - Parameter evaluations
  - `bayes_objective_plot.png` - Objective function landscape
  - `bayes_regret_plot.png` - Regret analysis

### 📈 **5_graphics.py**
Comprehensive final analysis and visualization suite for model interpretation.
- **Features**: Feature importance, permutation importance, correlation heatmaps, SHAP analysis
- **Advanced Visualizations**: Beeswarm plots, dependence plots, waterfall plots

## 🗂️ Output Directories

| Directory | Source | Contents |
|-----------|--------|----------|
| `📁 rs_graphs/` | Random Search | Parameter importance, confusion matrix |
| `📁 gs_graphs/` | Grid Search | Performance matrices |
| `📁 bs_graphs/` | Bayesian Search | Optimization plots, convergence analysis |
| `📁 analysis_graphs/` | Results Analysis | Comparative visualizations |

## 📋 Model Files

- `rs_model.pkl` - Best Random Search model
- `gs_model.pkl` - Best Grid Search model
- `bs_model.pkl` - Best Bayesian Search model

## 📊 Data Files

- `rs_dataframe.csv` - Random Search results
- `gs_dataframe.csv` - Grid Search results
- `bs_dataframe.csv` - Bayesian Search results

## 🚀 Usage

1. **Initial Exploration**: Run `1_rf_model_creator_random_search.py` for baseline
2. **Systematic Search**: Run `3_grid_search_optimizer.py` for comprehensive optimization
3. **Intelligent Optimization**: Run `4_bayes_search_optimizer.py` for Bayesian tuning
4. **Analysis**: Use `2_results_load_only.py` and `5_graphics.py` for detailed insights

## 🎯 Medical Application Notes

- **Primary Metric**: Recall (sensitivity) prioritized for medical diagnosis
- **Secondary Metric**: F1-score for balanced evaluation during optimization
- **Clinical Validation**: Always validate with domain experts and real patient data

---

*Built with scikit-learn, scikit-optimize, matplotlib, seaborn, and SHAP*