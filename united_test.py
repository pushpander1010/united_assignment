#Imports

import pandas as pd
import warnings
from helpers import *
import joblib
warnings.filterwarnings('ignore')

# Load
test_fares_df = pd.read_csv("test\\test\\test_fares_data.csv")
test_schedules_df = pd.read_csv("test\\test\\test_schedules.csv", index_col=0)
service_index_df = pd.read_csv("training\\training\\service_index.csv", index_col=0)

ids=test_fares_df.copy().pop('Unnamed: 0')
test_fares=test_fares_df.drop('Unnamed: 0',axis=1)

# Mappings
sched_train = process_schedule_data(test_schedules_df)
sched_test = process_schedule_data(test_schedules_df)
serv_map = process_service_data(service_index_df)

# Prepare
X_processed, _, fitted_imputer = prepare_data(test_fares_df,"", sched_train, serv_map)

# Load the final trained model
final_model = joblib.load('final_tuned_xgb_model.pkl')
# Predict on the processed test data
predictions = final_model.predict(X_processed)

predictions=pd.DataFrame({'Unnamed: 0':ids,'predicted_fares':predictions})

test_fares_data_pushpander_kumar=test_fares_df.merge(predictions, on='Unnamed: 0',how='left',validate='one_to_one')
test_fares_data_pushpander_kumar.to_csv("test_fares_data_pushpander_kumar.csv",index=False)

print(test_fares_data_pushpander_kumar.head(3))
print(test_fares_data_pushpander_kumar.shape,test_fares_df.shape)


