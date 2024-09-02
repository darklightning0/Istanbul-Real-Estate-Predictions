import pandas as pd
import tensorflow as tf   
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt

data = pd.read_csv('./datasets/model_input_dataset.csv')

inp = data.drop(columns=['price']).values
tar = data['price'].values

inp_train, inp_temp, tar_train, tar_temp = train_test_split(inp, tar, test_size=0.4, random_state=42)
inp_val, inp_test, tar_val, tar_test = train_test_split(inp_temp, tar_temp, test_size=0.5, random_state=42)

def build_model(hp):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(units=hp.Int('units_1', min_value=16, max_value=256, step=32),
                                    activation='sigmoid', input_shape=(inp_train.shape[1],)))
    
    model.add(tf.keras.layers.Dense(units=hp.Int('units_2', min_value=8, max_value=128, step=16),
                                    activation='sigmoid')),
    
    model.add(tf.keras.layers.Dense(units=hp.Int('units_3', min_value=16, max_value=256, step=32),
                                    activation='sigmoid')),
    
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(
                    hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_absolute_error',
                  metrics=['mae'])
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',  
    max_trials=10, 
    executions_per_trial=2, 
)

tuner.search(inp_train, tar_train, epochs=50, validation_data=(inp_val, tar_val), batch_size=1024)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best units for first layer: {best_hps.get('units_1')}")
print(f"Best units for second layer: {best_hps.get('units_2')}")
print(f"Best units for third layer: {best_hps.get('units_3')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")
