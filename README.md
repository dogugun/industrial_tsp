# Industrial Time Series Prediction
This project is intended to be a showcase in my personal portfolio. 

The goal is to estimate a target sensor value in a 12 hours of prediction horizon. Our inputs are 6 input sensors and our target values in 
the current time and in a 6 hours of window. 

The aim of the showcase was to make predictions with a recurrent neural network, more specifically an LSTM. We predict individual values in the 12 hours of prediction horizon. 
Our LSTM model is trained to minimize the mean squared error, this also drove me to measure performance of my model with root mean squared error. 

The exploratory data analysis, model training and extraction are done within the jupyter notebooks. The exported model is saved as json file and saved in models folder along with the weights file. 
That way, the model is ready for scoring. Since it is a pretty simple demo, the scoring is done on the same data with the training. But this could be easily replaced with 
predictions from a source system. 

The file and folder structure of the project is as follows. 
C:\Users\dozkaya\AppData\Local\Continuum\anaconda3\python.exe C:/Users/dozkaya/Projects/Finance/greta_ozel_harekat/portfolio_industrial_tsp/file_summary.py
```
 portfolio_industrial_tsp/
 ├── data/
 │   ├── processed/
 │   │   └── processed_data.csv
 │   ├── raw/
 │   │   └── time_series_data.xlsx
 │   └── results/
 │       └── scoring_results.csv
 ├── file_summary.py
 ├── industrial_tsp/
 │   ├── industrial_tsp/
 │   │   ├── config.py - configurable variables
 │   │   └── custom_funcs.py - util functions
 │   └── setup.py
 ├── models/
 │   ├── lstm_model.h5 - weights of the LSTM model
 │   └── lstm_model.json - serialized and saved LSTM model
 ├── notebooks/
 │   ├── eda_model_build.html - HMTL format of the notebook
 │   └── eda_model_build.ipynb - notebook for the EDA and model building
 ├── README.md
 └── scripts/
    ├── execute.py
    ├── preprocess.py - detrending, deseasoning the data, generating time window
    └── score.py - scoring with the loaded LSTM
```

The current project can be run directly by calling the execute script, or predict function in the score.py. The code structure is kept simple on purpose to be inspected with ease.
