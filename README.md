# Netflix-Find-Best-Combination

This project provides a Python function `find_best_combin()` that automatically evaluates combinations of preprocessing techniques and classification models to identify the best-performing setup based on user-defined evaluation metrics.

## Function: `find_best_combin(data, target_column, scoring_metrics=['accuracy', 'f1', 'roc_auc'])`

Automatically searches for the best combination of data preprocessing (scaling, encoding), classification models using cross-validation and multiple evaluation metrics.

### Parameters
- **data**: `pd.DataFrame`  
  Entire dataset with features and target.

- **target_column**: `str`  
  Name of the target column (for binary classification).

- **scoring_metrics**: `list of str`, default=`['accuracy', 'f1', 'roc_auc']`  
  List of metrics used to evaluate model performance.

### Attributes
The function returns:

- `best_model` (`dict`) — Best performing model configuration.
- `top_5` (`pd.DataFrame`) — Top 5 configurations sorted by combined score.

#### `best_model` keys:
- `model`: The name of the classification model used.
- `scaler`: The name of the scaler used for numeric features.
- `encoder`: The name of the encoder used for categorical features.
- `best_params`: A dictionary of hyperparameters selected by GridSearchCV.
- `accuracy`: Accuracy score on the test set.
- `f1`: F1 score on the test set.
- `roc_auc`: Area under the ROC curve (for binary classification with probability prediction).
- `combined_score`: Mean of all specified scoring metrics.

## Result Interpretation

- Both **Top 1 and Top 2** combinations achieved **identical scores**, with the **only difference being the scaler used**.
  - **Top 1** used: `StandardScaler`
  - **Top 2** used: `MinMaxScaler`
- This indicates that the model is **robust across different scaling techniques** and performs consistently regardless of the scaling method.

## Architecture

1. Load data (CSV)
2. Preprocess target variable (Movie vs TV Show)
3. For each combination of:
   - Scaler: `StandardScaler`, `MinMaxScaler`
   - Encoder: `OneHotEncoder`
   - Model: `DecisionTree`, `RandomForest`, `KNeighbors`
4. Build pipeline: `Preprocess → Model`
5. Perform `GridSearchCV` with 5-fold cross-validation
6. Evaluate each model with metrics: `accuracy`, `f1`, `roc_auc`
7. Sort all combinations by `combined_score` (mean of metrics)
8. Return `best_model` and `top_5` configurations
