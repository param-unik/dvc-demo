from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# Load in the data #
seed = 42
df = pd.read_csv("data/wine_quality.csv")

# Split into train and test sections
y = df.pop("quality")

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=seed
)

# Define the model
model = RandomForestRegressor(random_state=42)

# Define the parameter grids for RandomSearchCV and GridSearchCV
param_distributions = {
    "n_estimators": [int(x) for x in np.linspace(start=50, stop=500, num=10)],
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_features": [1.0, "sqrt", "log2"],
    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
    "min_samples_split": [int(x) for x in np.linspace(2, 10)],
    "min_samples_leaf": [int(x) for x in np.linspace(1, 10)],
}

param_grid = {
    "n_estimators": [50, 100, 200, 300, 400],
    "criterion": ["squared_error", "absolute_error"],
    "max_features": [1.0, "sqrt", "log2"],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 7, 9],
    "min_samples_leaf": [3, 5, 7, 9],
}

# Perform RandomSearchCV
random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=50,
    cv=5,
    random_state=42,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error",
)
random_search.fit(X_train, y_train)

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring="r2")
grid_search.fit(X_train, y_train)

print("Random search cv best params are : \n")
print(random_search.best_params_)
print("with best score: ", random_search.best_score_)

print("\nGrid search cv best params are : \n")
print(grid_search.best_params_)
print("with best score: ", grid_search.best_score_)


# Compare the best scores and select the best model
if (random_search.best_score_) * -1 > (grid_search.best_score_) * -1:
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    search_method = "RandomSearchCV"
else:
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    search_method = "GridSearchCV"

# Train the final model on the entire training data
final_model = best_model
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
test_score = r2_score(y_test, y_pred)

print(f"Best parameters found using {search_method}: {best_params}")
print(f"Test score of the final model: {test_score:.4f}")
