import cafcoding.constants as constants
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import logging 
logger = logging.getLogger(constants.LOGGER_ID)

def scaling_attributes(continuous_list, train, test, val=None):
    # initialize the column names of the continuous data

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    train[continuous_list] = cs.fit_transform(train[continuous_list])
    test[continuous_list] = cs.transform(test[continuous_list])
    if val is not None:
        val[continuous_list] = cs.transform(val[continuous_list])

    return train,test, val

def continous_column_detector(df):
    continuous = []
    for col in df.columns:
        try:
            min = np.min(df[col])
            max = np.max(df[col])

            if min >= 0 and max<=1:
                continue

            continuous.append(col)
        except TypeError as ex:
            logger.warning(f'{col} error: {str(ex)}')
        
    return continuous

def prepare_dataset(data, ignored_columns, target, columns_drops=None, list_ut=None, kfold=None):
    
    X={}
    y={}
    features={}

    if ignored_columns is None:
        ignored_columns = [] 

    if columns_drops is None:
        columns_drops = []

    if list_ut is not None:
        data = pd.concat([data[data.ut==ut] for ut in list_ut])
    
    X = data.drop(list(set(ignored_columns+columns_drops)), axis=1)#[target]
    features =X.columns
    X= X.values
    y=data[target].values
    if kfold is not None:
        X[kfold,:]
        y[kfold]
        return X[kfold,:],  y[kfold], features
    return X,  y, features

def create_model(input_dim, layers, dropout=None, batch_normalization = False, regress=False):
    # define our MLP network
    logger.info("Create model")
    logger.debug(f'input_dim: {input_dim}')
    logger.debug(f'layers: {layers}')
    logger.debug(f'dropout: {dropout}')
    logger.debug(f'batch_normalization: {batch_normalization}')
    logger.debug(f'regress: {regress}')

    dropout = 0.0 if dropout is None else dropout

    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, activation="relu"))
    for layer in layers[1:]:
        if dropout:
            model.add(Dropout(dropout))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Dense(layer, activation="relu"))

    if regress:
        model.add(Dense(1, activation="linear"))
    else:
        model.add(Dense(1, activation="sigmoid"))
    return model

