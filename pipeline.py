import os
import sys
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def main():
    try:
        
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        files_dir = os.path.join(base_dir, "Files")
        input_path = os.path.join(files_dir, "input.txt")
        output_path = os.path.join(files_dir, "output.txt")
        
        scaler_x_path = get_resource_path("modelss/scaler_selected_x.pkl")
        scaler_y_path = get_resource_path("modelss/scaler_selected_y.pkl")
        mlp_path     = get_resource_path("modelss/mlp_s_model.pkl")

        scaler_x      = joblib.load(scaler_x_path)
        scaler_y      = joblib.load(scaler_y_path)
        mlpS_pipeline = joblib.load(mlp_path)

        values = pd.read_csv(input_path, header=None).squeeze("columns").values

        common_npy = [
            'cubic', 'rd_b2', 'eleneg_b2', 'distance_b1', 'eleneg_b1',
            'ionenergy_b1', 'distance_b2', 'rs_b1', 'hoe_b2',
            'heat_of_formation', 'rp_b1'
        ]

        dff = pd.DataFrame([values], columns=common_npy)
        dff_x_scaled = pd.DataFrame(scaler_x.transform(dff), columns=common_npy)

        pred_scaled = mlpS_pipeline.predict(dff_x_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        output_dir  = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "output.txt")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            for v in pred:
                f.write(f"{v}\n")

        print("Order:\n", common_npy)
        print("Input:\n", dff, "\n")
        print("Predicted output:\n", pred)
        print()
        print(f"\nSaved in {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        
    input("Enter to exit")

if __name__ == "__main__":
    main()
