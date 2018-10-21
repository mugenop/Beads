import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def show_img(array):
    plt.imshow(array)

def compute(path_in_images,path_out_images,):

    seed = 73
    X = np.load(path_in_images)
    print(X.shape)

    profiles = np.zeros(shape=(X.shape[0], int(X.shape[2] / 2)), dtype=np.float)


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