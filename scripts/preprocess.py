import sys
sys.path.insert(1, '../industrial_tsp/industrial_tsp')
import pandas as pd
import config as config
import custom_funcs as funcs
import keras.models

raw_data_path = config.raw_data_path
sheet_name = config.sheet_name

df_sc = pd.read_excel(raw_data_path, sheet_name=sheet_name)
df_sc.columns = ['timestamp', 'target', 'input_1', 'input_2', 'input_3', 'input_4', 'input_5', 'input_6']

df_sc['input_2'] = funcs.detrend(df_sc['input_2'])
df_sc['input_2'] = funcs.difference(df_sc['input_2'])
df_sc['input_3'] = funcs.detrend(df_sc['input_3'])
df_sc['input_3'] = funcs.difference(df_sc['input_3'])



df_sc = funcs.set_vector_date(df_sc, config.vector_length)
df_sc = funcs.set_target_in_ph(df_sc, config.prediction_horizon)
df_sc.dropna(inplace=True)

x = df_sc[['target', 'input_1', 'input_2', 'input_3', 'input_4', 'input_5', 'input_6', 'target_t-1', 'target_t-2',
             'target_t-3', 'target_t-4', 'target_t-5', 'target_t-6', 'input_1_t-1', 'input_1_t-2', 'input_1_t-3',
             'input_1_t-4', 'input_1_t-5', 'input_1_t-6', 'input_2_t-1', 'input_2_t-2', 'input_2_t-3', 'input_2_t-4',
             'input_2_t-5', 'input_2_t-6', 'input_3_t-1', 'input_3_t-2', 'input_3_t-3', 'input_3_t-4', 'input_3_t-5',
             'input_3_t-6', 'input_4_t-1', 'input_4_t-2', 'input_4_t-3', 'input_4_t-4', 'input_4_t-5', 'input_4_t-6',
             'input_5_t-1', 'input_5_t-2', 'input_5_t-3', 'input_5_t-4', 'input_5_t-5', 'input_5_t-6', 'input_6_t-1',
             'input_6_t-2', 'input_6_t-3', 'input_6_t-4', 'input_6_t-5', 'input_6_t-6']]

x.to_csv(config.processed_data_path, index=False, header=True)
