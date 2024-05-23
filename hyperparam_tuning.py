import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


def perform_randomCV_tuning(X_features, y_labels):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

    # Number of features to consider at every split
    max_features = [None, "sqrt", "log2"]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]

    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=100,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    # Fit the random search model
    rf_random.fit(X_features, y_labels)

    print("Best Parameters: for RandomSearchCV \n")
    pprint(rf_random.best_params_)
    pprint("Best Score = {:.3f}".format(rf_random.best_score_))

    best_estimator_ = rf_random.best_estimator_

    return best_estimator_


def perform_GridSearchCV_tuning(X_features, y_labels):
    # Create the parameter grid based on the results of random search
    param_grid = {
        "bootstrap": [True, False],
        "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        "max_depth": [80, 90, 100, 110],
        "max_features": [1.0, "sqrt", "log2"],
        "min_samples_leaf": [3, 4, 5],
        "min_samples_split": [8, 10, 12],
        "n_estimators": [100, 200, 300, 400],
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        verbose=2,
    )
    # Fit the grid search to the data
    grid_search.fit(X_features, y_labels)
    print("Best Parameters: for GridSearchCV \n")
    pprint(grid_search.best_params_)
    pprint("Best Score = {:.3f}".format(grid_search.best_score_))

    best_estimator_ = grid_search.best_estimator_

    return best_estimator_


def evaluate(model, X_features, y_labels):
    predictions = model.predict(X_features)
    r2 = r2_score(y_labels, predictions) * 100

    print("Model Performance with R2 score is: \n")
    print("R2 Score is ", r2)

    return r2
