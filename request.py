import requests
import random
from model import pre_processing
from sklearn.preprocessing import StandardScaler


url = 'http://localhost:5000/predict'
payload = {}

df, feature_col, target_col_logistic = pre_processing()
testing_df_request = df.iloc[int(.8*df.shape[0]) : ]

random_int = random.randint(0,241)
random_row = testing_df_request[feature_col].iloc[random_int]

for name, val in zip(random_row.index, random_row.values):
    payload[name] = val
    
r =  requests.post(url,json=payload)

print(payload, random_int)
print("\n\n\n")
print(r.json())
print("\n")
