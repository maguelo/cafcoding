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
from tensorflow.keras.optimizers import Adam,RMSprop


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



    

def train_model(train_data, val_data, params, model_conf, filename = None):
    logger.debug('Train model - params: {}'.format(str(params)))
    logger.debug('Train model - model_conf: {}'.format(str(model_conf)))
                 
    X_train, y_train = train_data
    X_val, y_val = val_data
    model = create_model(X_train.shape[1], 
                         params['layers'], 
                         dropout=params.get('dropout',0.0), 
                         batch_normalization=params.get('batch_normalization',False),
                         regress=True)
    
    callback_list = []

    if  model_conf.get('early_stopping', None) is not None:
    # patient early stopping
        early_stopping = EarlyStopping(monitor=model_conf['early_stopping'].get('monitor','val_loss'), mode='min', verbose=model_conf['early_stopping'].get('verbose',1), patience=model_conf['early_stopping'].get('monitor_patience',15))
        callback_list.append(early_stopping)
    
    if filename is not None:
        logger.info(f'Checkpoint model in {filename}')
        model_checkpoint = ModelCheckpoint(filename, 
                                           monitor=model_conf['model_checkpoint'].get('monitor','val_loss'), 
                                           mode='min', 
                                           verbose=model_conf['model_checkpoint'].get('verbose',1), 
                                           save_best_only=model_conf['model_checkpoint'].get('save_best',True))
        
        callback_list.append(model_checkpoint)
    
    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our target *predictions* and the *actual prices*
    opt = RMSprop(lr=params["lr"], rho=0.9, epsilon=None, decay=params["decay"])
    #opt = Adam(lr=params["lr"], decay=params["decay"])
    model.compile(loss=model_conf['model'].get('loss_func',"mean_absolute_percentage_error"), 
                  optimizer=opt,
                  metrics=model_conf['model'].get('metrics',None))

    # train the model
    logger.info("[INFO] training model...")
    
    history =model.fit(x=X_train, y=y_train,
            callbacks=callback_list,
            validation_data=(X_val, y_val),
            epochs=model_conf.get('epochs',200), 
            batch_size=params.get("batch_size",64))

    return history, early_stopping.best
