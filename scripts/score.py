import sys

from tensorflow.python.keras.models import model_from_json

sys.path.insert(1, '../industrial_tsp/industrial_tsp')
import pandas as pd
import config as config
import custom_funcs as funcs
import keras.models
from sklearn.preprocessing import StandardScaler

def predict():
    json_file = open(config.model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(config.model_weights_path)

    processed_data = pd.read_csv(config.processed_data_path)
    processed_data = processed_data[['target','input_1','input_2','input_3','input_4','input_5','input_6','target_t-1','input_1_t-1','input_2_t-1','input_3_t-1','input_4_t-1','input_5_t-1','input_6_t-1','target_t-2','input_1_t-2','input_2_t-2','input_3_t-2','input_4_t-2','input_5_t-2','input_6_t-2','target_t-3','input_1_t-3','input_2_t-3','input_3_t-3','input_4_t-3','input_5_t-3','input_6_t-3','target_t-4','input_1_t-4','input_2_t-4','input_3_t-4','input_4_t-4','input_5_t-4','input_6_t-4','target_t-5','input_1_t-5','input_2_t-5','input_3_t-5','input_4_t-5','input_5_t-5','input_6_t-5','target_t-6','input_1_t-6','input_2_t-6','input_3_t-6','input_4_t-6','input_5_t-6','input_6_t-6']]

    scaler = StandardScaler()
    processed_data=scaler.fit_transform(processed_data)

    number_of_timesteps = config.vector_length + 1 # from t to t-6
    number_of_inputs = config.input_sensors+1 #+1 for the target itself at time t
    processed_data = processed_data.reshape(processed_data.shape[0], number_of_inputs, number_of_timesteps)

    result = loaded_model.predict(processed_data)

    pd.DataFrame(result).to_csv(config.results_data_path, index=False, header=False)

