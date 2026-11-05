# packages
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import joblib



#read csv
df_sleep_score = pd.read_csv("C:/Users/zains/Downloads/AML project/clean_realistic_survey_data_500k.csv")

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

#scaling to make mean=0 and sd=1 for cognitive assessment score, should help nn work better
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# xgboost post Y scaling
xgb2 = XGBRegressor(
     n_estimators=927,
     learning_rate=0.01698468634711624,
     max_depth=6,
     min_child_weight=8,
     subsample=0.6186338996381091,
     colsample_bytree=0.7379898350863366,
     gamma=0.13994843655998704,
     reg_alpha=2.2469365057047685e-06,
     reg_lambda=0.0007592610979206302,
     random_state=42,
     n_jobs=-1
 )

xgb2.fit(X_train_encoded, y_train_scaled)
y_pred_xgb2 = xgb2.predict(X_test_encoded)

#testing scaled vs unscaled(
xgb2 = XGBRegressor(
    n_estimators=927,
    learning_rate=0.01698468634711624,
    max_depth=6,
    min_child_weight=8,
    subsample=0.6186338996381091,
    colsample_bytree=0.7379898350863366,
    gamma=0.13994843655998704,
    reg_alpha=2.2469365057047685e-06,
    reg_lambda=0.0007592610979206302,
    random_state=42,
    n_jobs=-1
)

# Train model using scaled target
xgb2.fit(X_train_encoded, y_train_scaled.ravel())

# Predict scaled values
y_pred_xgb2_scaled = xgb2.predict(X_test_encoded)

# Convert predictions back to original scale
y_pred_xgb2 = y_scaler.inverse_transform(
    y_pred_xgb2_scaled.reshape(-1, 1)
).ravel()

print("SCALED RESULTS")
print("R²:", r2_score(y_test_scaled, y_pred_xgb2_scaled))
print("MSE:", mean_squared_error(y_test_scaled, y_pred_xgb2_scaled))
print("RMSE:", root_mean_squared_error(y_test_scaled, y_pred_xgb2_scaled))
# SCALED RESULTS
# R²: 0.624716106159647
# MSE: 0.375939592183241
# RMSE: 0.6131391295482951

joblib.dump({
    "model": xgb2,
    "preprocessor": preprocessor,
    "y_scaler": y_scaler,
    "feature_columns": X.columns.tolist()
}, "xgboost_sleep_model.joblib")

print("Model successfully saved!")
