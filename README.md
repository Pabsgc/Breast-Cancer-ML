# 🔬 Machine Learning Pipeline for Breast Cancer Detection

A streamlined machine learning pipeline for breast cancer diagnosis using a Random Forest classifier with multiple hyperparameter optimization strategies: Random Search, Grid Search, and Bayesian Optimization.

## 📁 Project Structure

### � **Audio Preprocessing Pipeline**

#### **0_audio_preprocessing.py**
Converts raw audio files to WAV format (8kHz, 16bit, Mono) using FFmpeg for consistent audio processing.
- **Input**: `audio_input/` (mp4, mpeg, ogg files)
- **Output**: `audio_output/` (wav files)

#### **1_audio_diarization.py**
Performs speaker diarization using pyannote.audio and calculates SNR (Signal-to-Noise Ratio) for audio quality classification.
- **Input**: `audio_output/` (wav files)
- **Output**:
  - `audio_output/speaker_out_1/` - First speaker audio segments
  - `audio_output/speaker_out_2/` - Second speaker audio segments
  - `audio_output/valid/` - High quality audio (SNR ≥ 15 dB)
  - `audio_output/marginal/` - Marginal quality audio (10 ≤ SNR < 15 dB)
  - `audio_output/invalid/` - Low quality audio (SNR < 10 dB)

#### **2_feature_extraction.py**
Extracts eGeMAPS v2.0 features from manually selected audio files using openSMILE, labeling each entry accordingly.
- **Input**: `audio_openSMILE/0/` (non-urgent class), `audio_openSMILE/1/` (urgent class)
- **Output**: `dataset/feature_extraction_dataset.csv`

---

### 🤖 **Model Training Pipeline**

#### **3_rf_model_creator_random_search.py**
Performs initial model training using **RandomizedSearchCV** for broad hyperparameter exploration. Includes 5-fold cross-validation and evaluates performance with accuracy, precision, recall, F1-score, and confusion matrix.
- **Inputs**: `dataset/feature_extraction_dataset.csv`
- **Outputs**:
  - `models/rs_model.pkl`
  - `dataframes/rs_dataframe.csv`
  - `graphs/rs_graphs/`
    - `rs_parameter_importance.png` 
    - `rs_confusion_matrix.png` 

#### **4_grid_search_optimizer.py**
Uses GridSearchCV for exhaustive hyperparameter tuning with systematic evaluation across the parameter grid.
- **Inputs**: `dataset/feature_extraction_dataset.csv`
- **Outputs**:
  - `models/gs_model.pkl`
  - `dataframes/gs_dataframe.csv`
  - `graphs/gs_graphs/`
    - `gs_confusion_matrix.png` 

#### **5_bayes_search_optimizer.py**
Applies Bayesian optimization (scikit-optimize) for efficient hyperparameter tuning using a Gaussian Process. The workflow transitions from random exploration to recall-focused optimization.
- **Inputs**: `dataset/feature_extraction_dataset.csv`
- **Outputs**:
  - `models/bs_model.pkl`
  - `dataframes/bs_dataframe.csv`
  - `graphs/bs_graphs/`
    - `bs_confusion_matrix.png` 
    - `bayes_convergence_plot.png` 
    - `bayes_evaluations_plot.png` 
    - `bayes_objective_plot.png` 
    - `bayes_regret_plot.png` 

---

### 📊 **Analysis & Visualization**

#### **6_graphics.py**
Generates advanced visual analysis for model interpretability, including feature importance, correlations, and SHAP-based explanations.
- **Inputs**: Trained model files, dataset
- **Outputs**: 
  - `graphs/analysis_graphs/`
    - `feature_importance.png` 
    - `permutation_importance.png` 
    - `correlation_heatmap.png` 
    - `pairplot.png` 
    - `beeswarm_plot.png` 
    - `force_plot.html` 
    - `dependence_plot.png` 
    - `waterfall.png` 

#### **7_results_load_only.py**
Loads previously trained models and generates comparative visualizations, including confusion matrices and parameter importance plots. Its purpose is purely for convenience.
- **Outputs**: `graphs/preloaded_graphs/`

---

## 🗂️ Outputs Overview

| File/Directory | Description |
|----------------|-------------|
| `audio_output/` | Processed audio files |
| `dataset/feature_extraction_dataset.csv` | Extracted features dataset |
| `models/rs_model.pkl` | Random Search model |
| `models/gs_model.pkl` | Grid Search model |
| `models/bs_model.pkl` | Bayesian Search model |
| `dataframes/rs_dataframe.csv` | Random Search results |
| `dataframes/gs_dataframe.csv` | Grid Search results |
| `dataframes/bs_dataframe.csv` | Bayesian Search results |
| `graphs/rs_graphs/` | Random Search visualizations |
| `graphs/gs_graphs/` | Grid Search visualizations |
| `graphs/bs_graphs/` | Bayesian Search visualizations |
| `graphs/preloaded_graphs/` | Preloaded model visualizations |
| `graphs/analysis_graphs/` | Final interpretability & insights |

## 🚀 Usage Workflow

### Step 1: Audio Preprocessing
1. Place raw audio files in `audio_input/`
2. Run **Audio Preprocessing** → `python 0_audio_preprocessing.py`
3. Run **Speaker Diarization** → `python 1_audio_diarization.py`
4. Organize processed audio:
   - Copy valid/marginal audio to `audio_openSMILE/0/` (non-urgent class)
   - Copy valid/marginal audio to `audio_openSMILE/1/` (urgent class)

### Step 2: Feature Extraction
5. Run **Feature Extraction** → `python 2_feature_extraction.py`

### Step 3: Model Training
6. Run **Random Search** → `python 3_rf_model_creator_random_search.py`
7. Run **Grid Search** → `python 4_grid_search_optimizer.py` (Very slow! Bayesian Search is recommended instead.)
8. Run **Bayesian Optimization** → `python 5_bayes_search_optimizer.py`

### Step 4: Analysis
9. Analyze results → `python 6_graphics.py`
10. View preloaded results → `python 7_results_load_only.py`

## 🎯 Medical Considerations

- **Recall (Sensitivity)** is prioritized to minimize false negatives
- Models should always be validated with clinical expertise and real-world data

---

*Built with scikit-learn, scikit-optimize, matplotlib, seaborn, SHAP, openSMILE, and pyannote.audio*
- **F1-Score** is used during optimization for balanced evaluation
- Models should always be validated with clinical expertise and real-world data

---

*Built with scikit-learn, scikit-optimize, matplotlib, seaborn, and SHAP*