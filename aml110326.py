# packages
import numpy as np
import pandas as pd
import sklearn as sk
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
# import torch
# import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
# import optuna




#read csv
df_sleep_score = pd.read_csv("C:/Users/zains/Downloads/AML project/clean_realistic_survey_data_500k.csv")
#understand the dataset structure and values
# print(df_sleep_score.shape[1])
# print(df_sleep_score.head())
# print(df_sleep_score.tail)
# print(df_sleep_score.shape)
# print("columns:", df_sleep_score.columns)
# print(df_sleep_score.dtypes)
# print(df_sleep_score.describe())

#checking score types
# print(df_sleep_score["cognitive_assessment_score"].unique())
# print(df_sleep_score["chronic_stress_score"].unique())
# print(df_sleep_score["social_connection_score"].unique())

#cognitive_assessment_score will be considered continuous, other two are ordinal


#check for nulls and duplicates
# print(df_sleep_score.isnull().sum())
# print(df_sleep_score.duplicated().sum())
#0 for both

#check min and max for impossible values
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 200)
# print(df_sleep_score.describe().T)
#both min and max are good

#check "smoking status" for incorrect strings
# print(df_sleep_score['smoking_status'].value_counts())
# print(df_sleep_score['smoking_status'].unique())
#only 3 values, all correct

#dataset is clean

# #visualization
# df_sleep_score.hist(figsize=(15, 10), bins=30)
# plt.tight_layout()
# plt.show()

#cognitive_assessment_score vs other variables to see any clear relationship
# sns.pairplot(df_sleep_score, y_vars=["cognitive_assessment_score"], x_vars=df_sleep_score.columns.drop("cognitive_assessment_score"))
# plt.show()

#train/test split
X = df_sleep_score.drop("cognitive_assessment_score", axis=1)
y = df_sleep_score["cognitive_assessment_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#changing smoking status into a numeric entity
preprocessor = ColumnTransformer(
    transformers=[
        ("smoking", OneHotEncoder(handle_unknown="ignore"), ["smoking_status"])
    ],
    remainder="passthrough"
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
# print(X_train_encoded.shape)
# print(preprocessor.get_feature_names_out())

#issue with more than 15 columns solves when realized onehotencoder split smoking status into 3 columns

#checking clean status
# print("status after splitting")
# print(X_train.isna().sum())
# print(X_train.dtypes)
# print(X_train["smoking_status"].value_counts())
# print(X_train.duplicated().sum())
# print(y_train.describe())
# print(y_train.min(), y_train.max())

#linear regression model
# lrmodel = LinearRegression()
# lrmodel.fit(X_train_encoded, y_train)
# y_pred_lr = lrmodel.predict(X_test_encoded)
# print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
# print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
#low R-squared value, relationship likely not linear

#ridge regression model
# ridge = Ridge(alpha=1.0)
#
# ridge.fit(X_train_encoded, y_train)
#
# y_pred_R = ridge.predict(X_test_encoded)
#
# print("Ridge R²:", r2_score(y_test, y_pred_R))
# print("Ridge MSE:", mean_squared_error(y_test, y_pred_R))
#also low R-squared value

#decision tree model
# tree = DecisionTreeRegressor(max_depth= 10, random_state=42)
#
# tree.fit(X_train_encoded, y_train)
#
# y_pred_tree = tree.predict(X_test_encoded)
#
# print("Decision tree R²:", r2_score(y_test, y_pred_tree))
# print("Decision tree MSE:", mean_squared_error(y_test, y_pred_tree))
#checking depth, error in depth caused by onehotencoder mistake
# print(tree.get_depth())
# print(tree.get_n_leaves())

#issue with 0.066 R-squared value came from onehotencoder, having only it predict cognitive_assessment_score

#random forest
# rf = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42,
#     n_jobs=-1
# )
#
# rf.fit(X_train_encoded, y_train)
#
# y_pred_rf = rf.predict(X_test_encoded)
#
# print("Random Forest R²:", r2_score(y_test, y_pred_rf))
# print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

#gradient boost
# gb = GradientBoostingRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=42
# )
#
# gb.fit(X_train_encoded, y_train)
#
# y_pred_gb = gb.predict(X_test_encoded)
#
# print("Gradient Boosting R²:", r2_score(y_test, y_pred_gb))
# print("Gradient Boosting MSE:", mean_squared_error(y_test, y_pred_gb))

#knn
# knn = make_pipeline(
#     StandardScaler(),
#     KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
# )
# knn.fit(X_train_encoded, y_train)
# print("kNN Test R²:", r2_score(y_test, knn.predict(X_test_encoded)))



#XGBoost
# xgb = XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.3,
#     max_depth=6,
#     random_state=42,
#     n_jobs=-1
# )
#
# xgb.fit(X_train_encoded, y_train)

#feature importance, seeing which values xgb leans on most
# feature_names = preprocessor.get_feature_names_out()
#
# importance = pd.Series(
#     xgb.feature_importances_,
#     index=feature_names
# ).sort_values(ascending=False)
#
# print(importance)
# columns with less than 1% importance potential to remove:
# sleep_quality_score
# screen_time_recreational_hrs
# diet_omega3_freq
# alcohol_drinks_wk
# social_media_hrs
# water_intake_liters
# sleep_duration_hrs
# smoking_status_Former



# y_pred_xgb = xgb.predict(X_test_encoded)
#
# print("XGBoost R²:", r2_score(y_test, y_pred_xgb))
# print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
# print("RMSE:", root_mean_squared_error(y_test, y_pred_xgb))

#hyperparameter tuning
# params = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [3, 6, 9],
#     "learning_rate": [0.03, 0.1, 0.2]
# }
#
# grid = GridSearchCV(
#     XGBRegressor(random_state=42),
#     params,
#     cv=5,
#     scoring="r2",
#     n_jobs=-1
# )
#
# grid.fit(X_train_encoded, y_train)
#
# print("Best parameters:", grid.best_params_)
# print("Best CV R²:", grid.best_score_)
#
# best_xgb = grid.best_estimator_
#
# y_pred_best = best_xgb.predict(X_test_encoded)
#
# print("\nTuned XGBoost R²:", r2_score(y_test, y_pred_best))
# print("Tuned XGBoost MSE:", mean_squared_error(y_test, y_pred_best))
#Best parameters: {'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 300}

#visual to see models prediction
# plt.scatter(y_test, y_pred_xgb, alpha=0.3)
#
# plt.xlabel("Actual Cognitive Assessment Score")
# plt.ylabel("Predicted Cognitive Assessment Score")
# plt.title("XGBoost: Actual vs Predicted")
#
# # Perfect prediction line
# plt.plot(
#     [y_test.min(), y_test.max()],
#     [y_test.min(), y_test.max()]
# )
#
# plt.show()

#check for overfitting
# y_pred_train_xgb = xgb.predict(X_train_encoded)
# y_pred_test_xgb = xgb.predict(X_test_encoded)
# print("XGBoost Train R²:", r2_score(y_train, y_pred_train_xgb))
# print("XGBoost Test R²: ", r2_score(y_test, y_pred_test_xgb))

#neural network, mlp model

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_encoded)
# X_test_scaled = scaler.transform(X_test_encoded)
#made R2 value much worse

#scaling to make mean=0 and sd=1 for cognitive assessment score, should help nn work better
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
#increased R2 value by 0.87
#after epoch changing and input size/layer changing and scaling:
# MSE: 0.418773353099823
# R²: 0.5819570422172546
# RMSE: 0.6471269992048106

#change train/test split into tensors
# X_train_nn = torch.tensor(X_train_encoded, dtype=torch.float32)
# X_test_nn = torch.tensor(X_test_encoded, dtype=torch.float32)
#
# y_train_nn = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)
# y_test_nn = torch.tensor(y_test_scaled, dtype=torch.float32).reshape(-1, 1)

# print(X_train_nn.shape, y_train_nn.shape)
# print(X_test_nn.shape, y_test_nn.shape)
# print(y_train_nn.min(), y_train_nn.max())

#define model
# class SleepMLP(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#         #increasing input size and hidden layers increased R2 by 0.4
#
#     def forward(self, x):
#         return self.network(x)
#
# #training componenets
# model = SleepMLP(X_train_nn.shape[1])
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# #training loop
# epochs = 300
# #changing epochs from 100 to 200 increased R2 by 1.25, 200 to 300 increased by 0.06
#
# for epoch in range(epochs):
#     model.train()
#
#     predictions = model(X_train_nn)
#     loss = criterion(predictions, y_train_nn)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
#
# #R and MSE
# model.eval()
#
# with torch.no_grad():
#     y_pred_nn = model(X_test_nn)
#
# y_pred_nn = y_pred_nn.numpy()
# y_actual_nn = y_test_nn.numpy()
#
# mse = mean_squared_error(y_actual_nn, y_pred_nn)
# r2 = r2_score(y_actual_nn, y_pred_nn)
#
# print("MSE:", mse)
# print("R²:", r2)
# print("RMSE:", mse ** 0.5)

#results, pre tuning
# Epoch 10, Loss: 152.3500
# Epoch 20, Loss: 70.2433
# Epoch 30, Loss: 22.8645
# Epoch 40, Loss: 15.6009
# Epoch 50, Loss: 14.6172
# Epoch 60, Loss: 12.9236
# Epoch 70, Loss: 12.0699
# Epoch 80, Loss: 11.2450
# Epoch 90, Loss: 10.5839
# Epoch 100, Loss: 9.9651
# MSE: 10.021782875061035
# R²: -2.080423355102539
# RMSE: 3.1657199615665683
#clear issue with preproccesing or training, negative R value indicated prediction is worse than simply just using average

# plot for model
# plt.figure(figsize=(8, 6))
#
# plt.scatter(y_actual_nn, y_pred_nn, alpha=0.4)

# plt.plot(
#     [y_actual_nn.min(), y_actual_nn.max()],
#     [y_actual_nn.min(), y_actual_nn.max()],
#     'r--'
# )
#
# plt.xlabel("Actual Sleep Score")
# plt.ylabel("Predicted Sleep Score")
# plt.title("MLP: Actual vs Predicted Sleep Scores")
#
# plt.show()
# #
#
# # xgboost post Y scaling
# xgb2 = XGBRegressor(
#      n_estimators=927,
#      learning_rate=0.01698468634711624,
#      max_depth=6,
#      min_child_weight=8,
#      subsample=0.6186338996381091,
#      colsample_bytree=0.7379898350863366,
#      gamma=0.13994843655998704,
#      reg_alpha=2.2469365057047685e-06,
#      reg_lambda=0.0007592610979206302,
#      random_state=42,
#      n_jobs=-1
#  )

# xgb2.fit(X_train_encoded, y_train_scaled)
# y_pred_xgb2 = xgb2.predict(X_test_encoded)
#
# #testing scaled vs unscaled(
# xgb2 = XGBRegressor(
#     n_estimators=927,
#     learning_rate=0.01698468634711624,
#     max_depth=6,
#     min_child_weight=8,
#     subsample=0.6186338996381091,
#     colsample_bytree=0.7379898350863366,
#     gamma=0.13994843655998704,
#     reg_alpha=2.2469365057047685e-06,
#     reg_lambda=0.0007592610979206302,
#     random_state=42,
#     n_jobs=-1
# )

# Train model using scaled target
# xgb2.fit(X_train_encoded, y_train_scaled.ravel())
#
# # Predict scaled values
# y_pred_xgb2_scaled = xgb2.predict(X_test_encoded)
#
# # # Convert predictions back to original scale
# # y_pred_xgb2 = y_scaler.inverse_transform(
# #     y_pred_xgb2_scaled.reshape(-1, 1)
# # ).ravel()

# print("SCALED RESULTS")
# print("R²:", r2_score(y_test_scaled, y_pred_xgb2_scaled))
# print("MSE:", mean_squared_error(y_test_scaled, y_pred_xgb2_scaled))
# print("RMSE:", root_mean_squared_error(y_test_scaled, y_pred_xgb2_scaled))

# # Original-scale metrics
# print("\nORIGINAL SCALE RESULTS")
# print("R²:", r2_score(y_test, y_pred_xgb2))
# print("MSE:", mean_squared_error(y_test, y_pred_xgb2))
# print("RMSE:", root_mean_squared_error(y_test, y_pred_xgb2))
# #)


# print("XGBoost#2 R²:", r2_score(y_test_scaled, y_pred_xgb2))
# print("XGBoost#2 MSE:", mean_squared_error(y_test_scaled, y_pred_xgb2))
# print("RMSE#2:", root_mean_squared_error(y_test_scaled, y_pred_xgb2))
#using scaled/normalized Y allowed us to significantly reduce MSE and RMSE
# XGBoost#2 R²: 0.6131601236429274 vs XGBoost R²: 0.6129363775253296
# XGBoost#2 MSE: 0.3875157653841628 vs XGBoost MSE: 1.259264588356018
# RMSE#2: 0.6225076428319277 vs RMSE: 1.1221696138381958
#after optuna's hyperparameters
#XGBoost#2 R²: 0.624716106159647
# XGBoost#2 MSE: 0.375939592183241
# RMSE#2: 0.6131391295482951

#hyperparameter tuning with optuna
# def objective(trial):
#
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 0.5),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
#         'random_state': 42,
#         'n_jobs': 1
#     }
#
#     model = XGBRegressor(**params)
#
#     scores = cross_val_score(
#         model,
#         X_train_encoded,
#         y_train_scaled.ravel(),
#         cv=3,
#         scoring='neg_root_mean_squared_error',
#         n_jobs=3
#     )
#
#     return -scores.mean()
#
#
# study = optuna.create_study(direction='minimize')
#
# study.optimize(objective, n_trials=50)
#
# print("Best Parameters:", study.best_params)
# print("Best CV RMSE:", study.best_value)
#results:
#Best Parameters: {'n_estimators': 927, 'learning_rate': 0.01698468634711624, 'max_depth': 6, 'min_child_weight': 8, 'subsample': 0.6186338996381091, 'colsample_bytree': 0.7379898350863366, 'gamma': 0.13994843655998704, 'reg_alpha': 2.2469365057047685e-06, 'reg_lambda': 0.0007592610979206302}
# Best CV RMSE: 0.6108462244789532