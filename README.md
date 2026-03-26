# 🔬 Random Forest Classifier for Breast Cancer Detection

A comprehensive machine learning pipeline for breast cancer diagnosis using Random Forest classification with multiple hyperparameter optimization strategies.

## 📁 Project Structure

### 🔍 **1_rf_model_creator_random_search.py**
Creates Random Forest model using RandomizedSearchCV for initial hyperparameter exploration.
- **Features**: 5-fold cross-validation, confusion matrix, accuracy scores, precision/recall/F1 metrics
- **Outputs**:
  - Model: `rs_model.pkl`
  - Dataframe: `rs_dataframe.csv`
  - Graphs: `📁 rs_graphs/`
    - `rs_parameter_importance.png` - Hyperparameter impact visualization
    - `rs_confusion_matrix.png` - Model performance matrix

### 📊 **2_results_load_only.py**
Loads pre-trained models and generates analysis graphics for model comparison.
- **Features**: Parameter importance plots, confusion matrices, performance metrics
- **Outputs**: Comparative visualizations in respective graph folders

### 🎯 **3_grid_search_optimizer.py**
Utilizes GridSearchCV (exhaustive search) for systematic hyperparameter optimization.
- **Features**: Monte Carlo-style parameter grid exploration, comprehensive evaluation
- **Outputs**:
  - Model: `gs_model.pkl`
  - Dataframe: `gs_dataframe.csv`
  - Graphs: `📁 gs_graphs/`
    - `gs_confusion_matrix.png` - Grid search performance matrix

### 🧠 **4_bayes_search_optimizer.py**
Implements Bayesian optimization using scikit-optimize for intelligent hyperparameter tuning.
- **Features**: Gaussian Process surrogate modeling, acquisition functions, convergence plots
- **Optimization Strategy**: RandomSearchCV → F1 exploration → BayesSearchCV → Recall optimization
- **Outputs**:
  - Model: `bs_model.pkl`
  - Dataframe: `bs_dataframe.csv`
  - Graphs: `📁 bs_graphs/`
    - `bayes_convergence_plot.png` - Optimization convergence
    - `bayes_evaluations_plot.png` - Parameter evaluations
    - `bayes_objective_plot.png` - Objective function landscape
    - `bayes_regret_plot.png` - Regret analysis

### 📈 **5_graphics.py**
Comprehensive final analysis and visualization suite for model interpretation.
- **Features**: Feature importance, permutation importance, correlation heatmaps, SHAP analysis
- **Advanced Visualizations**: Beeswarm plots, dependence plots, waterfall plots, force plots
- **Outputs**: `📁 analysis_graphs/`
  - `feature_importance.png` - Feature importance bar plot
  - `permutation_importance.png` - Permutation importance analysis
  - `correlation_heatmap.png` - Top features correlation heatmap
  - `pairplot.png` - Pairwise feature relationships
  - `beeswarm_plot.png` - SHAP summary plot
  - `force_plot.html` - Interactive SHAP force plot
  - `dependence_plot.png` - SHAP dependence plot
  - `waterfall.png` - SHAP waterfall plot

## 🗂️ Output Files and Directories

| File/Directory | Description |
|----------------|-------------|
| `rs_model.pkl` | Trained Random Search model |
| `gs_model.pkl` | Trained Grid Search model |
| `bs_model.pkl` | Trained Bayesian Search model |
| `rs_dataframe.csv` | Random Search results dataframe |
| `gs_dataframe.csv` | Grid Search results dataframe |
| `bs_dataframe.csv` | Bayesian Search results dataframe |
| `📁 rs_graphs/` | Random Search visualizations |
| `📁 gs_graphs/` | Grid Search visualizations |
| `📁 bs_graphs/` | Bayesian Search visualizations |
| `📁 analysis_graphs/` | Final model analysis graphics |

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