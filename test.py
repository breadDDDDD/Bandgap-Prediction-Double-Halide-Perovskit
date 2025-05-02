import numpy as np
import pandas as pd
import joblib
import os

url_input = 'Files/input.txt'
url_output = 'Files/output.txt'

scaler_x = joblib.load('modelss/scaler_selected_x.pkl')
scaler_y = joblib.load('modelss/scaler_selected_y.pkl')
mlpS_pipeline = joblib.load('modelss/mlp_s_model.pkl')

common_npy = [ 'cubic', 'rd_b2', 'eleneg_b2', 'distance_b1' ,'eleneg_b1' ,'ionenergy_b1',
 'distance_b2', 'rs_b1' ,'hoe_b2' ,'heat_of_formation' ,'rp_b1'
]

values = pd.read_csv(url_input, header=None).squeeze("columns").values
dff = pd.DataFrame([values], columns=common_npy)

dff_x_scaled = pd.DataFrame(scaler_x.transform(dff), columns=common_npy)

predS_pipeline_scaled = mlpS_pipeline.predict(dff_x_scaled)
predS_pipeline = scaler_y.inverse_transform(predS_pipeline_scaled.reshape(-1, 1)).ravel()

os.makedirs(os.path.dirname(url_output), exist_ok=True)
with open(url_output, "w") as f:
    for value in predS_pipeline:
        f.write(f"{value}\n")

print("order:\n", common_npy)
print("input:\n", dff)
print("Scaled input:\n", dff_x_scaled)
print("Predicted output:\n", predS_pipeline)
