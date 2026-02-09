#Imports

import pandas as pd
import warnings
from helpers import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler,KBinsDiscretizer,PowerTransformer
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_predict,cross_val_score
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline  
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_predict,cross_val_score
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,StackingRegressor,VotingRegressor,HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet,PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin

# Load
service_index_df = pd.read_csv("training\\training\\service_index.csv", index_col=0)
train_fares_df = pd.read_csv("training\\training\\train_fares.csv", index_col=0, on_bad_lines='skip')
train_schedules_df = pd.read_csv("training\\training\\train_schedules.csv", index_col=0)
test_fares_df = pd.read_csv("test\\test\\test_fares_data.csv")
test_schedules_df = pd.read_csv("test\\test\\test_schedules.csv", index_col=0)

y = train_fares_df.pop('total_fare')

# Mappings
sched_train = process_schedule_data(train_schedules_df)
sched_test = process_schedule_data(test_schedules_df)
serv_map = process_service_data(service_index_df)

# Prepare
X_processed, y_processed, fitted_imputer = prepare_data(train_fares_df, y, sched_train, serv_map)
X_test_processed, _, _ = prepare_data(test_fares_df, pd.Series(0, index=test_fares_df.index), sched_test, serv_map, imputer=fitted_imputer)

# Split & Train
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Pipeline
num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

# Preprocess
exclude_cols = {
    'route_freq',
    'flight_duration_log',
    'flights_per_route_day',
    'share_by_demand',
    'carrier_flights_route_day',
    'carrier_share_route_day',
    'share_x_congestion',
    'days_until_departure',
    'flight_duration_mean',
    'scaled_demand',
    'scaled_share',
    'tz_mean',
    'mo',
    'day_of_week',
}

other_num_cols = [c for c in num_cols if c not in exclude_cols]


num_ops=ColumnTransformer(
    transformers=[
         ('binning',KBinsDiscretizer(n_bins=5,strategy='kmeans'),['route_freq','flight_duration_log','flights_per_route_day',]),
         ('power_tr',PowerTransformer(standardize=True),[
         'share_by_demand','carrier_flights_route_day','carrier_share_route_day',
         'flights_per_route_day','share_x_congestion','days_until_departure',
         'days_until_departure',
         'flight_duration_mean','scaled_demand','scaled_share']),
    ])
num_ops_pipe=Pipeline(
    steps=[
        ('num_ops',num_ops)
    ]
)

numeric_pipe = Pipeline(steps=[
    ('scale',StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=50, sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num_ops',num_ops_pipe,num_cols),
        ('num', numeric_pipe, other_num_cols),
        ('cat', categorical_pipe, cat_cols),
    ],
    remainder='passthrough',n_jobs=-1
)

best_params = {
    'subsample': 0.9,
    'reg_lambda': 2,
    'reg_alpha': 1,
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42,
}

final_xgb = XGBRegressor(**best_params)

final_pipe = Pipeline(steps=[
    ('prep', preprocess),
    ('model', final_xgb)
])

# Fit on the Training Data
print("Training final tuned XGBoost model...")
final_pipe.fit(X_train, y_train)

# Predict and Calculate Metrics
y_pred = final_pipe.predict(X_val)

r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = root_mean_squared_error(y_val, y_pred)

print("   FINAL MODEL PERFORMANCE")
print("="*30)
print(f"Validation R2   : {r2:.4f}")
print(f"Validation MAE  : {mae:.4f}")
print(f"Validation RMSE : {rmse:.4f}")

# Save the Final Model
joblib.dump(final_pipe, "final_tuned_xgb_model.pkl")
print("\nFinal model saved as final_tuned_xgb_model.pkl")