# Random Forest Classifier for Breast Cancer Detection

## File descriptions:

### 1_rf_model_creator_random_search.py

Creates Random Forest model using RandomSearch. Prints graphics for consequent hyperparameter tuning. Uses 5 folds to create a Confussion Matrix and print Accuracy Scores. Also prints Precision, Recall and F1 scores. 

### 2_results_load_only.py

Prints the same results as the script above, preloading already created models. 

### 3_grid_search_optimizer.py 

Utilises the GridSearch method (MonteCarlo/BruteForce) to optimize the model's hyperparameters. 

### 4_bayes_search_optimizer.py

Utilises the BayesSearch method from scikit.optimize to optimize the model's hyperparameters. It will probably be best practice to first do a RandomSearchCV to locate the highest F1-Score areas of the hyperparameter space, and subsequently do a BayesSearchCV to locate the hyperparameter combination that results in the best possible Recall Score.

### 5_graphics.py:

Prints final analyisis, graphs and results corresponding to the desired model/DatFrame.