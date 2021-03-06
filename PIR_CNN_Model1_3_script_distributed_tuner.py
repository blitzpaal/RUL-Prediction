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

    data_train = data_train_df[data_train_df.ID <= 1000]
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


class MyHyperModel(HyperModel):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self, hp):
        # build model
        input_layer = Input(shape=self.input_shape)

        x = LayerNormalization(axis=1)(input_layer)
        x = Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1))(x)

        for i in range(hp.Int('num_layers', min_value=2, max_value=8, step=2)):
            x = Conv1D(filters=hp.Int('filters', min_value=20, max_value=50, step=5),
                    kernel_size=hp.Int('kernel_size', min_value=3, max_value=12, step=3),
                    padding="same",
                    activation='relu')(x)

        x = Flatten()(x)

        x = Dense(self.output_shape, activation='softmax')(x)
        output_layer = x

        model = Model(input_layer, output_layer)

        # compile model
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
            loss='sparse_categorical_crossentropy',
            metrics=['SparseCategoricalAccuracy'])

        return model


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
    hypermodel = MyHyperModel(input_shape = (sequence_length, 3), output_shape = 20)

tuner = RandomSearch(
    hypermodel,
    objective='val_sparse_categorical_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='CNN_Model1_3',
    project_name='1000_structures_randomSearch_1')

tuner.search_space_summary()

tuner.search(train_dataset,
             epochs=500,
             verbose=2,
             validation_data=val_dataset,
             callbacks=[tf.keras.callbacks.TensorBoard(tuner.directory + '/' + tuner.project_name)])

models = tuner.get_best_models(num_models=1)

tuner.results_summary()

best_model = models[0]

model_path = tuner.directory + '/' + tuner.project_name + '/' + 'best_model'

# get model as json string and save to file
model_as_json = best_model.to_json()
with open(model_path + '.json', "w") as json_file:
    json_file.write(model_as_json)
# save model weights
best_model.save_weights(model_path + '_weights.h5')