# Random Forest Classifier for Breast Cancer Detection

## File descriptions:

### 1_rf_model_creator_random_search.py

Creates Random Forest model using RandomSearch. Prints graphics for consequent hyperparameter tuning. Uses 5 folds to create a Confussion Matrix and print Accuracy Scores. Also prints Precision, Recall and F1 scores. 

### 2_results_load_only.py

Prints the same results as the script above, preloading already created models. 

### 3_grid_search_optimizer.py 

Utilises the GridSearch method (MonteCarlo/BruteForce) to optimize the model's hyperparameters. 

### 4_bayes_search_optimizer.py

Utilises the BayesSearch method from scikit.optimize to optimize the model's hyperparameters. 

### 5_graphics.py:

Prints final analyisis, graphs and results corresponding to the desired model/DatFrame.