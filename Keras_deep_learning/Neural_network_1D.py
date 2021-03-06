import os
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D,MaxPooling1D,AveragePooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers.core import Dense
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adam,SGD
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import pickle
import os


def mono_layered_CNN(height):
    model = Sequential()
    inputShape = (height, 1)
    chanDim = -1
    model.add(Conv1D(filters=32, kernel_size=2, padding="same",
                     input_shape=inputShape, data_format='channels_last'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv1D(filters=32, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv1D(filters=32, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=64, kernel_size=2, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv1D(filters=64, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv1D(filters=64, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=128, kernel_size=2, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    # model.add(Conv1D(filters=128, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    #
    # model.add(Conv1D(filters=128, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))

    # model.add(MaxPooling1D(pool_size=2, strides=2))
    # model.add(Dropout(0.25))
    # model.add(Conv1D(filters=128, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))

    # model.add(Conv1D(filters=32, kernel_size=2, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(AveragePooling1D(pool_size=2, strides=2))
    # model.add(Dense(1024))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    # softmax classifier
    model.add(Dense(1))
    # model.add(Activation("linear"))

    # return the constructed network architecture
    return model

def scorer(y_test,y_predict):

    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.4f"
          % mean_squared_error(y_test, y_predict))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.4f' % r2_score(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

def create_model2(height):
    model = Sequential()
    inputShape = (height, 1)
    chanDim = -1
    model.add(Dense(512,
                    input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dense(512, ))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))

    # model.add(Dense(64))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Dense(64))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Dense(64))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    #
    # model.add(Dropout(0.25))
    #
    # model.add(Dense(128))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    #
    # model.add(Dense(128))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    #
    # model.add(Dense(128))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    #
    # model.add(Dropout(0.25))
    #
    # model.add(Dense(128))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    #
    # model.add(Dense(32))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1))

    # return the constructed network architecture
    return model

def compute(path_in_data,path_in_target,path_in_snr, path_in_z,path_images, path_model_out,path_fig_out):

    list_images = os.listdir(path_images)
    X = np.load(path_in_data)
    h = X.shape[1]
    X = X.reshape(X.shape[0],  X.shape[1],1)
    Z_labels = np.load(path_in_target).astype(np.float32)
    Z_base = np.load(path_in_z).astype(np.float32)
    SNR = np.load(path_in_snr)
    l = int(Z_labels.shape[0]/Z_base.shape[0])
    for i in range(len(list_images)):
        Z_labels[i*l:(i+1)*l] = Z_base[i]
    # index = np.where(Z_labels>=-0.001)
    # Z_labels = Z_labels[index]
    # X = X[index]
    D = 10
    X = X[::D]
    Z_labels = Z_labels [::D]
    SNR = SNR[::D]
    num_cores = 8

    if GPU:
        num_GPU = 1
        num_CPU = 1
    else:
        num_GPU = 0
        num_CPU = 1
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,inter_op_parallelism_threads=num_cores, device_count = {'CPU' : num_CPU, 'GPU' : num_GPU},allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    session = tf.Session(config=config)
    K.set_session(session)
    indices = np.arange(0, X.shape[0], 1)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Z_labels, indices,
                                                                                     test_size=0.20)
    SNR_test = SNR[indices_test]



    # initialize the model
    print("[INFO] compiling model...")
    model = mono_layered_CNN(h)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS, momentum=0.9, nesterov=True)
    if LOAD:
        model.load_weights(path_model_out)

        model.compile(loss="mean_squared_error", optimizer=opt,metrics=["mean_squared_error"])
        y_predict = model.predict(X_test.astype(np.float32))
    else:
        print("[INFO] training network...")
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])
        checkpointer=ModelCheckpoint(filepath=path_model_out,monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min',)
        reducelronplateau = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.5, patience=10, verbose=0, mode='min', min_delta=0.001, cooldown=0, min_lr=0.0001)
        earlyStop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.001, patience=20, verbose=0, mode='min', baseline=None, restore_best_weights=False)
        # training the model
        print ('Training Start....')
        hist = model.fit( X_train.astype(np.float32), y_train.astype(np.float32), batch_size=BS, nb_epoch=EPOCHS,
                      verbose=1, validation_data=(X_test.astype(np.float32), y_test),callbacks=[checkpointer,reducelronplateau,earlyStop])
        model.load_weights(path_model_out)

        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])
        y_predict = model.predict(X_test.astype(np.float32))

    scorer(y_test,y_predict)
    arr1inds = y_test.argsort()

    y_test = y_test[arr1inds[::-1]]
    y_predict = y_predict[arr1inds[::-1]].reshape(y_predict.shape[0],)
    SNR_test = SNR_test[arr1inds[::-1]]
    # save the model to disk
    print("[INFO] Show results...")
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    deltay = y_test - y_predict

    plt.plot(y_test, '-o', color='darkorange', label='data')
    plt.plot(y_predict, color='darkblue', label='prediction')
    plt.plot(SNR_test * 10, '--^', label='SNR')
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(path_fig_out)
    plt.show()

if __name__ == '__main__':
    EPOCHS = 100
    INIT_LR = 1e-3
    BS = 32
    path_images = "C:/Users/Mugen/PycharmProjects/Keras_deep_learning/Images/"
    path_in = "G:/Keras/"
    path_model = path_in + "model_1D_MonoLayered.h5"
    path_save_fig = path_in + "model_output_1D_MonoLayered.png"
    GPU = True
    LOAD = False
    path_in_p = path_in+ "profiles_lessNoise.npy"
    path_in_z = path_in+ "z_lessNoise.npy"
    path_in_snr = path_in+ "snr_lessNoise.npy"
    path_z = path_in + "zs.npy"

    compute(path_in_p,path_in_z,path_in_snr,path_z, path_images,path_model,path_save_fig)