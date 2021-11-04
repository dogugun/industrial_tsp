# config.py

from pathlib import Path

raw_data_dir = Path('../data/raw')
raw_data_path = raw_data_dir / 'time_series_data.xlsx'

results_data_dir = Path('../data/results')
results_data_path = results_data_dir / 'scoring_results.csv'

processed_data_dir = Path('../data/processed')
processed_data_path = processed_data_dir / 'processed_data.csv'

model_dir = Path('../models')
model_path = model_dir / 'lstm_model.json'
model_weights_path = model_dir / 'lstm_model.h5'

sheet_name='veriseti'

numeric_cols=['target','input_1','input_2','input_3','input_4','input_5','input_6']

prediction_horizon = 12
vector_length = 6
input_sensors = 6