#pip install keras-tcn --no-dependencies

# imports
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import pickle
import h5py

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from tcn import *

from datetime import datetime

# set folder path
folder = 'data'
fd = folder
fd_km = fd

# set random seed
np.random.seed(1)
tf.random.set_seed(1)

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
#utils 
nb_gauges = 3
data_train = data_train_df[data_train_df.ID <= 1000]
data_val = data_train_df[data_train_df.ID > 9900]

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
        yield data_array[stop-1, :]#data_array[start+1:stop+1, :]
   

#prepare data
seq_cols =  ['gauge'+str(i) for i in range(1,4)]#['label'+str(i) for i in range(1,4)]
seq_cols1 =  ['RUL_bin']
sequence_length = 20
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

# build model
input_layer = Input(shape=(dbX.shape[1], dbX.shape[2]))
x = TCN(nb_filters=20, kernel_size=3, nb_stacks=1, dilations=[2 ** i for i in range(6)], padding='causal',
            use_skip_connections=True, dropout_rate=0.1, return_sequences=False,
            activation='relu', kernel_initializer='he_normal', use_batch_norm=False, use_layer_norm=False,
            use_weight_norm=True, name='TCN')(input_layer)
x = Dense(nb_bins)(x)
x = Activation('softmax')(x)
output_layer = x
model = Model(input_layer, output_layer)

# compile model
opt = Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train model
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d%m%Y%H%M%S")

model_path = 'models/PIR_CNN_Model3_5_' + timestampStr

# get model as json string and save to file
model_as_json = model.to_json()
with open(model_path + '.json', "w") as json_file:
    json_file.write(model_as_json)

# save initial weights
model.save_weights(model_path + '_weights.h5')

es = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=50, verbose=0, mode='max')
mc = keras.callbacks.ModelCheckpoint(model_path + '_weights_epoch_{epoch}.h5', monitor='accuracy', mode='max', 
                                     save_weights_only=True, save_best_only=False)

history = model.fit(dbX, dbY, epochs=50, batch_size=32,
        validation_data=(dbX_val, dbY_val), verbose=2, callbacks = [es,mc])

# save learning history
np.save(model_path + '_history.npy', history.history)