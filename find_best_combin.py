from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

def find_best_combin(data, target_column, scoring_metrics=['accuracy', 'f1', 'roc_auc']):
    """
    Automatically evaluates multiple preprocessing and model combinations
    to find the best classification pipeline based on given scoring metrics.

    Parameters:
    - data (pd.DataFrame): The input dataset including features and the target column.
    - target_column (str): The name of the target column for binary classification.
    - scoring_metrics (list of str): A list of metrics to evaluate model performance. 
      Supported metrics include 'accuracy', 'f1', and 'roc_auc'.

    Returns:
    - best_model (dict): A dictionary containing the best performing model configuration and scores.
    - top_5 (pd.DataFrame): A DataFrame listing the top 5 combinations sorted by combined score.
    """
    results = []

    # 1. Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numeric and categorical features
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    cat_features = X.select_dtypes(include='object').columns.tolist()

    # 2. Define scalers and encoders
    scalers = [StandardScaler(), MinMaxScaler()]
    encoders = [OneHotEncoder(handle_unknown='ignore')]

    # 3. Define models and hyperparameters
    models = {
        'DecisionTree': (DecisionTreeClassifier(), {
            'model__max_depth': [3, 5, 10],
            'model__min_samples_split': [2, 5]
        }),
        'RandomForest': (RandomForestClassifier(), {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10]
        }),
        'KNeighbors': (KNeighborsClassifier(), {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance']
        })
    }

    # Iterate over all combinations of scaler, encoder, and model
    for scaler in scalers:
        for encoder in encoders:
            # Preprocessing pipeline for numerical and categorical features
            preprocessor = ColumnTransformer(transformers=[
                ('num', scaler, num_features),
                ('cat', encoder, cat_features)
            ])
            for model_name, (model, params) in models.items():
                # Combine preprocessing and model into a pipeline
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])

                # Use GridSearchCV with 5-fold cross-validation
                grid = GridSearchCV(pipe, param_grid=params, scoring='accuracy', cv=KFold(n_splits=5), n_jobs=-1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                grid.fit(X_train, y_train)

                # Predict and evaluate the model
                y_pred = grid.predict(X_test)
                y_prob = grid.predict_proba(X_test)[:, 1] if hasattr(grid, "predict_proba") else None

                result = {
                    'model': model_name,
                    'scaler': scaler.__class__.__name__,
                    'encoder': encoder.__class__.__name__,
                    'best_params': grid.best_params_,
                }

                # Compute evaluation metrics
                for metric in scoring_metrics:
                    if metric == 'accuracy':
                        result['accuracy'] = accuracy_score(y_test, y_pred)
                    elif metric == 'f1':
                        result['f1'] = f1_score(y_test, y_pred)
                    elif metric == 'roc_auc' and y_prob is not None and len(np.unique(y_test)) == 2:
                        result['roc_auc'] = roc_auc_score(y_test, y_prob)
                results.append(result)

    # Sort results by average score across selected metrics
    results_df = pd.DataFrame(results)
    results_df['combined_score'] = results_df[scoring_metrics].mean(axis=1)
    results_df = results_df.sort_values(by='combined_score', ascending=False)

    return results_df.iloc[0].to_dict(), results_df.head(5)

def main():
    # Load Netflix dataset
    data = pd.read_csv('netflix_titles.csv')

    # Convert 'type' to binary target: Movie = 1, TV Show = 0
    data = data[data['type'].isin(['Movie', 'TV Show'])]
    data['target'] = (data['type'] == 'Movie').astype(int)
    data = data.drop(columns=['type'])

    # Run AutoML pipeline and get top results
    target_column = 'target'
    best_model, top_5 = find_best_combin(data, target_column)

    print("\nBest Model:")
    print(f"Model           : {best_model['model']}")
    print(f"Scaler          : {best_model['scaler']}")
    print(f"Encoder         : {best_model['encoder']}")
    print(f"Best Parameters : {best_model['best_params']}")
    print(f"Accuracy        : {best_model['accuracy']:.4f}")
    print(f"F1 Score        : {best_model['f1']:.4f}")
    print(f"ROC_AUC         : {best_model['roc_auc']:.4f}")
    print(f"Combined Score  : {best_model['combined_score']:.4f}")
    
    print("\nTop 5 Models:")
    print(top_5)

if __name__ == "__main__":
    main()
