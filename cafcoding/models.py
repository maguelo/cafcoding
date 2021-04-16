# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Reshape
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# from keras.callbacks import EarlyStopping, ModelCheckpoint




# def create_model(input_dim, layers, regress=False):
#     # define our MLP network
#     model = Sequential()
#     model.add(Dense(layers[0], input_dim=input_dim, activation="relu"))
#     for layer in layers[1:]:
#         model.add(Dense(layer, activation="relu"))
#     if regress:
#         model.add(Dense(1, activation="linear"))
#     else:
#         model.add(Dense(1, activation="sigmoid"))
#     return model

    

# def prepare_model(params, train_data, val_data=None,validation_split=0.2, filename = None):
#     X_train,X_train_img, y_train = train_data
    
#     use_val_data = False
#     if val_data is not None:
#         X_val,X_val_img, y_val = val_data
#         use_val_data =True
    
    
#     model = create_model(train_data.shape[1], layers=params["layers"])
    
#    # patient early stopping
#     early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    
#     callback_list = [early_stopping]
    
#     if filename is not None:
#         print ("Checkpoint model in ", filename)
#         model_checkpoint = ModelCheckpoint(filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
#         callback_list.append(model_checkpoint)
    
#     # compile the model using mean absolute percentage error as our loss,
#     # implying that we seek to minimize the absolute percentage difference
#     # between our price *predictions* and the *actual prices*
#     opt = Adam(lr=params["lr"][0], decay=params["lr"][1])
#     model.compile(loss="mean_squared_error", optimizer=opt)

#     # train the model
#     print("[INFO] training model...")
    
#     if use_val_data:
#         history =model.fit(x=[X_train,X_train_img], y=y_train,
#                        callbacks=callback_list,
#                        validation_data=([X_val, X_val_img], y_val),
#                        epochs=params["epochs"], batch_size=params["batch_size"])

#     else:
#         history =model.fit(x=[X_train, X_train_img], y=y_train,
#                        callbacks=callback_list,
#                        validation_split=validation_split,
#                        epochs=params["epochs"], batch_size=params["batch_size"])
#     return history, early_stopping.best
