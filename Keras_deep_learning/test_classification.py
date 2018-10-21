import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, MaxPooling2D,InputLayer, Input, Conv2D, Flatten
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from time import time
def map (a):
    max = np.amax(a)
    min = np.amin(a)
    alpha = 255./(max-min)
    beta =255-alpha*max
    a_new =alpha*a+beta
    return a_new.astype(dtype = np.uint8 )
t1=time()
images = np.load('images.npy')
images_255 = map(images)

t2=time()
print (images_255[0])
print('Shape: ',images.shape)
print('Time took to read: '+str(t2-t1) +"seconds.")
t1=time()
z = np.load('zs.npy')
t2=time()
print('Shape: ',z.shape)
print('Time took to read: '+str(t2-t1) +"seconds.")
path = "C:/Users/Mugen/PycharmProjects/Keras_deep_learning/Images/"

for i in range(images.shape[0]):
    cv2.imwrite(path+'image_'+str(i)+'.png', images_255[i])
