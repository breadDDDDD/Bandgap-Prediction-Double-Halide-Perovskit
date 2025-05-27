# Energy Bandgap on Double Halide Perovskites Analysis, Machine Learning Approach on Bandgap Predictions

In this experiment, data on 540 double halide perovskites will be used to predict the crystal's bandgap using various machine learning models, where feature selection will be used to identify features with high correlation to the bandgap and be applied in training the model.


Data used in this experiment originated from a online material repository, where that data available are based on DFT calculations. Due to the data, the prediction made by the model will also be based on DFT calculations. DFT underestimates the bandgap in their calculation, so the prediction done will also have it's bandgap underestimated.
The dataset after being cleaned is present in this repository as 2halide.csv

## Analysis
Inside the JupyterNotebook, the analysis of model selection, data preparations, and selection of features are present. The model used in the application file is built and designed inside this notebook. Since the task is relatively straightforward and we're testing different models, we opted to use a Scikit-learn-based approach. All analysis done using and for the model is done within this notebook.

## Application
Inside the dist folder, an application file named pipeline.exe exist. This file contains the model and necessary processing needed for the model to run. To predict bandgaps of double halide perovskite using the application, save the data of inputs in the files/input.txt. For the input file, it must consist of 11 features that has been selected from this experiment, those features are ['cubic' 'rd_b2' 'eleneg_b2' 'distance_b1' 'eleneg_b1' 'ionenergy_b1'
 'distance_b2' 'rs_b1' 'hoe_b2' 'heat_of_formation' 'rp_b1'], where the order of inputting the file must be the same. 
 
 After saving the values in the text file, click the application file (pipeline.exe) and the bandgap prediction will be available in the output.txt file inside the folder and on the terminal display. An example of how to include the data inside input.txt is already present within the file, to use it just replace the data with the data of the structure that wants to be predicted
