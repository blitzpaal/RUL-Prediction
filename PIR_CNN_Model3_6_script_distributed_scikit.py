# Import
import numpy as np
import pandas as pd
import json
#from mpi4py import MPI
import random

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from kerastuner.tuners import *
from kerastuner import HyperModel
from keras.wrappers.scikit_learn import KerasClassifier
from tcn import *

from sklearn.model_selection import GridSearchCV

from datetime import datetime

# set random seed
np.random.seed(1)
tf.random.set_seed(1)


#prepare forecasting data
def gen_X_sequence(id_df, seq_length, seq_cols, timesteps_pred, type_data = None):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """

    ind_start = 0
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0+ind_start, num_elements-seq_length+1-timesteps_pred), range(seq_length+ind_start, num_elements+1-timesteps_pred)):
        yield data_array[start:stop, :]
 

def gen_Y_sequence(id_df, seq_length, seq_cols, timesteps_pred, type_data = None):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """

    ind_start = 0
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0+ind_start, num_elements-seq_length+1-timesteps_pred), range(seq_length+ind_start, num_elements+1-timesteps_pred)):
        yield data_array[stop-1, :]

   
def get_dataset(sequence_length, batch_size):
    # set folder path
    folder = 'data'
    fd = folder
    fd_km = fd

    # import data
    data_train_df = pd.read_pickle(fd_km + '/data_train_v1').reset_index().iloc[:,1:] #full set
    data_test_df = pd.read_pickle(fd_km + '/data_test_v1').reset_index().iloc[:,1:]

    # create bins
    l = 0.5
    nb_bins = 20 # including one extra bin for RUL>upper_bin_bound
    lower_bin_bound = 0
    upper_bin_bound = 80000

    bins = np.linspace(lower_bin_bound**l, upper_bin_bound**l, nb_bins)**(1/l)
    bins = np.append(bins, data_train_df.RUL.max())

    labels=[i for i in range(bins.shape[0]-1)]

    # categorise data
    data_train_df['RUL_bin'] = pd.cut(data_train_df['RUL'], bins=bins, labels=labels)
    data_test_df['RUL_bin'] = pd.cut(data_test_df['RUL'], bins=bins, labels=labels)

    # build data sequences
    #data_train_group = [data_train_df for _, data_train_df in data_train_df.groupby('ID')]
    #random.shuffle(data_train_group)
    #data_train_df_random = pd.concat(data_train_group)

    data_train = data_train_df[data_train_df.ID <= 100]
    data_val = data_train_df[data_train_df.ID > 9900]
    data_test = data_test_df

    #prepare data
    seq_cols = ['gauge'+str(i) for i in range(1,4)]
    seq_cols1 = ['RUL_bin']
    timesteps_pred = 1

    #training set
    seq_gen = (list(gen_X_sequence(data_train[data_train['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'train')) 
                    for id in data_train['ID'].unique())
    # generate sequences and convert to numpy array
    dbX = np.concatenate(list(seq_gen))

    seq_gen = (list(gen_Y_sequence(data_train[data_train['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'train')) 
                    for id in data_train['ID'].unique())
    # generate sequences and convert to numpy array
    dbY = np.concatenate(list(seq_gen)).reshape(-1,)

    #val set
    seq_gen = (list(gen_X_sequence(data_val[data_val['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'train')) 
                    for id in data_val['ID'].unique())
    # generate sequences and convert to numpy array
    dbX_val = np.concatenate(list(seq_gen))

    seq_gen = (list(gen_Y_sequence(data_val[data_val['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'train')) 
                    for id in data_val['ID'].unique())
    # generate sequences and convert to numpy array
    dbY_val = np.concatenate(list(seq_gen)).reshape(-1,)

    #test set
    seq_gen = (list(gen_X_sequence(data_test[data_test['ID']==id], sequence_length, seq_cols, timesteps_pred, type_data= 'train')) 
                    for id in data_test['ID'].unique())
    # generate sequences and convert to numpy array
    dbX_test = np.concatenate(list(seq_gen))

    seq_gen = (list(gen_Y_sequence(data_test[data_test['ID']==id], sequence_length, seq_cols1, timesteps_pred, type_data= 'train')) 
                    for id in data_test['ID'].unique())
    # generate sequences and convert to numpy array
    dbY_test = np.concatenate(list(seq_gen)).reshape(-1,)

    return (
        tf.data.Dataset.from_tensor_slices((dbX, dbY)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((dbX_val, dbY_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((dbX_test, dbY_test)).batch(batch_size),
    )


def get_compiled_model():
    input_shape = (30, 3)
    output_shape = 20

    # build model
    input_layer = Input(shape=input_shape)
    x = TCN(nb_filters=30, kernel_size=5,
                nb_stacks=1, dilations=[2 ** i for i in range(6)], padding='causal',
                use_skip_connections=True, dropout_rate=0.1, return_sequences=False,
                activation='relu', kernel_initializer='he_normal', use_batch_norm=False, use_layer_norm=False,
                use_weight_norm=True, name='TCN')(input_layer)
    x = Dense(output_shape)(x)
    x = Activation('softmax')(x)
    output_layer = x

    model = Model(input_layer, output_layer)

    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['SparseCategoricalAccuracy'])

    return model


def get_saved_compiled_model():
    # load model from file
    loaded_json = open('models/PIR_CNN_Model3_5_20052021100914.json', 'r').read()
    loadedModel = model_from_json(loaded_json, custom_objects={'TCN': TCN})

    tcn_full_summary(loadedModel, expand_residual_blocks=False)

    # restore weights
    loadedModel.load_weights('models/PIR_CNN_Model3_5_20052021100914_weights_epoch_500.h5')

    # compile model
    opt = Adam(learning_rate=0.0001)
    loadedModel.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])

    return loadedModel


# Load training, validation and test data
batch_size = 4096
sequence_length = 30
train_dataset, val_dataset, test_dataset = get_dataset(
    sequence_length=sequence_length, batch_size=batch_size)


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model_CV = KerasClassifier(build_fn=get_compiled_model, epochs=5, 
                           batch_size=batch_size, verbose=2)

    model = get_compiled_model()


# train model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%m%Y%H%M%S")

model_path = 'models/PIR_CNN_Model3_6_' + timestampStr

# get model as json string and save to file
model_as_json = model.to_json()
with open(model_path + '.json', "w") as json_file:
    json_file.write(model_as_json)

# save initial weights
model.save_weights(model_path + '_weights.h5')

es = keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0, patience=50, verbose=0, mode='max')
mc = keras.callbacks.ModelCheckpoint(model_path + '_weights_epoch_{epoch}.h5', monitor='sparse_categorical_accuracy', mode='max', 
                                     save_weights_only=True, save_best_only=False)

# define the grid search parameters
kernel_size = [2, 4, 8, 16]

param_grid = dict(kernel_size=kernel_size)

grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=3)

# Load training, validation and test data
train_dataset, val_dataset, test_dataset = get_dataset(sequence_length=30, batch_size=batch_size)

# run model
grid_result = grid.fit(train_dataset, validation_data=val_dataset, callbacks = [es,mc])

# print results
print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f' mean={mean:.#4}, std={stdev:.4} using {param}')

# save learning history
np.save(model_path + '_history.npy', history.history)

# Test the model on all available devices.
model.evaluate(test_dataset, verbose=2)