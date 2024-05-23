import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from hyperparam_tuning import (
    perform_GridSearchCV_tuning,
    perform_randomCV_tuning,
    evaluate,
)

# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data #
df = pd.read_csv("wine_quality.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=seed
)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(random_state=seed)

from pprint import pprint

# Look at parameters used by our current forest
print("Parameters currently in use:\n")
pprint(regr.get_params())


regr_randomCV = perform_randomCV_tuning(X_features=X_train, y_labels=y_train)
regr_gridCV = perform_GridSearchCV_tuning(X_features=X_train, y_labels=y_train)

train_r2score_random = evaluate(
    model=regr_randomCV, X_features=X_train, y_labels=y_train
)
test_r2score_random = evaluate(model=regr_randomCV, X_features=X_test, y_labels=y_test)

print("RandomSearchCV tuned Model's output is: \n")
print("Test R2 Score is ", train_r2score_random)
print("Train R2 score is ", test_r2score_random)


train_r2score_grid = evaluate(model=regr_gridCV, X_features=X_train, y_labels=y_train)
test_r2score_grid = evaluate(model=regr_gridCV, X_features=X_test, y_labels=y_test)

print("GridSearchCV tuned Model's output is: \n")
print("Test R2 Score is ", train_r2score_grid)
print("Train R2 score is ", test_r2score_grid)

# Write scores to a file
try:
    with open("metrics.txt", "w") as outfile:
        outfile.write(
            "Training R2 score from RandomSearchCV tuned model is : %2.1f%%\n"
            % train_r2score_random
        )
        outfile.write(
            "Test R2 score from RandomSearchCV tuned model is : %2.1f%%\n"
            % test_r2score_random
        )

        outfile.write(
            "Training R2 score from GridSearchCV tuned model is : %2.1f%%\n"
            % train_r2score_grid
        )
        outfile.write(
            "Test R2 score from GridSearchCV tuned model is : %2.1f%%\n"
            % test_r2score_grid
        )
        print("File metrics.txt gets created...")
except Exception as e:
    print("Exception happend during file write as ", str(e))


##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################

# Calculate feature importance in random forest from RandomSearchCV tuned Model
importances = regr_randomCV.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(
    list(zip(labels, importances)), columns=["feature", "importance"]
)
feature_df = feature_df.sort_values(
    by="importance",
    ascending=False,
)

# image formatting
axis_fs = 18  # fontsize
title_fs = 22  # fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel("Importance", fontsize=axis_fs)
ax.set_ylabel("Feature", fontsize=axis_fs)  # ylabel
ax.set_title(
    "Random forest's feature importance \n from RandomSearchCV tuned model",
    fontsize=title_fs,
)

plt.tight_layout()
plt.savefig("feature_importance_random.png", dpi=120)
plt.close()

# Calculate feature importance in random forest from RandomSearchCV tuned Model
importances = regr_gridCV.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(
    list(zip(labels, importances)), columns=["feature", "importance"]
)
feature_df = feature_df.sort_values(
    by="importance",
    ascending=False,
)

# image formatting
axis_fs = 18  # fontsize
title_fs = 22  # fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel("Importance", fontsize=axis_fs)
ax.set_ylabel("Feature", fontsize=axis_fs)  # ylabel
ax.set_title(
    "Random forest's feature importance \n from GridSearchCV tuned model",
    fontsize=title_fs,
)

plt.tight_layout()
plt.savefig("feature_importance_grid.png", dpi=120)
plt.close()
print("feature_importance_grid.png got created!")

##########################################
############ PLOT RESIDUALS  #############
##########################################

y_pred = regr_randomCV.predict(X_test) + np.random.normal(0, 0.25, len(y_test))
y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter, y_pred)), columns=["true", "pred"])

ax = sns.scatterplot(x="true", y="pred", data=res_df)
ax.set_aspect("equal")
ax.set_xlabel("True wine quality", fontsize=axis_fs)
ax.set_ylabel("Predicted wine quality", fontsize=axis_fs)  # ylabel
ax.set_title("Residuals per RandomSearchCV tuned model", fontsize=title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], "black", linewidth=1)
plt.ylim((2.5, 8.5))
plt.xlim((2.5, 8.5))

plt.tight_layout()
plt.savefig("residuals_random.png", dpi=120)
plt.close()
print("residuals_random.png got created!")


y_pred = regr_gridCV.predict(X_test) + np.random.normal(0, 0.25, len(y_test))
y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter, y_pred)), columns=["true", "pred"])

ax = sns.scatterplot(x="true", y="pred", data=res_df)
ax.set_aspect("equal")
ax.set_xlabel("True wine quality", fontsize=axis_fs)
ax.set_ylabel("Predicted wine quality", fontsize=axis_fs)  # ylabel
ax.set_title("Residuals per GridSearchCV tuned model", fontsize=title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], "black", linewidth=1)
plt.ylim((2.5, 8.5))
plt.xlim((2.5, 8.5))

plt.tight_layout()
plt.savefig("residuals_grid.png", dpi=120)
plt.close()
print("residuals_grid.png got created!")

print("train.py got completely successfully!")
