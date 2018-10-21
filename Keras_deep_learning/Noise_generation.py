"""Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance."""


import numpy as np
import os
import cv2
from time import time
from matplotlib import pyplot as plt
def SNR_measure(x,y):
    rho = np.mean((x-np.mean(x))*(y-np.mean(y)))/(np.sqrt(np.mean((x-np.mean(x))**2))*np.sqrt(np.mean((y-np.mean(y))**2)))
    # print(rho)
    # snr = 10*np.log10(rho**2/(1-rho**2))
    return rho**2
def make_profile(img,th0=1.):
    theta = np.arange(-np.pi, np.pi, np.pi / th0)
    R = 0
    i = 0
    x0 = 25
    y0 = 25
    rad = np.arange(0, 25, 1)
    for th in theta:
        i += 1

        xc = rad * np.cos(th)
        x = x0 + xc
        x[x < 0] = 0
        yc = rad * np.sin(th)
        y = y0 + yc
        profil = img[y.astype(int), x.astype(int)] / 255.
        R += profil

    # plt.plot(R / theta.shape[0],'-o')
    # plt.pause(1)
    # plt.close()
    return rad /np.amax(rad), R / theta.shape[0]
def gaussian_noise(image,mean=0,sigma=0.3):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy
def salt_pepper_noise(image,amount = 0.005,s_vs_p=0.5):
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

def poisson_noise(image,offset =0.01 ):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))/offset
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle_noise(image,kernel=5.):
    row,col = image.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)
    noisy = image + image * gauss * kernel
    return noisy


path = "C:/Users/Mugen/PycharmProjects/Keras_deep_learning/Images/"
path_out = "G:/Keras/"

images = os.listdir(path)
range_gauss = np.arange(0.00015,0.15015,0.00015)
range_salt_pepper = np.arange(0.000005,0.005+0.000005,0.000005)
range_poisson_noise = np.arange(0.0005,0.5+0.0005,0.0005)
range_speckle_noise = np.arange(0.00001,0.01+0.00001,0.00001)
print(range_gauss.shape)
print(range_salt_pepper.shape)
print(range_poisson_noise.shape)
print(range_speckle_noise.shape)
size = len(images)*(1+range_gauss.shape[0]+range_salt_pepper.shape[0]+range_poisson_noise.shape[0]+range_speckle_noise.shape[0])
output_path_images = path_out+"images_lessNoise.npy"
output_path_snr = path_out+"snr_lessNoise.npy"
output_path_z = path_out+"z_lessNoise.npy"

img = cv2.imread(path + images[0], 0)
a = np.zeros(shape=(size,img.shape[0],img.shape[1]),dtype = np.uint8)
k = 0
z = np.zeros(shape=(size),dtype = np.float64)
snr = np.zeros(shape=(size),dtype = np.float64)
p = np.zeros(shape=(size,int(img.shape[0]/2.)),dtype = np.uint8)
for i in range(len(images)):
    print("Treating image : " + str(i) + "/" + str(len(images)+1))
    img = cv2.imread(path + images[i], 0)
    a[k] = img
    # _,p[k] = make_profile(img,th0=180.)
    z[k] = i
    k+=1
    img = img.astype(np.float64)
    snr[k] = SNR_measure(img, img)
    print("Gaussian")
    t1 = time()
    for val in range_gauss:
        try:
            gauss = gaussian_noise(img,sigma=val)
        except ValueError:
            gauss = gaussian_noise(img, sigma=val)
        snr[k] = SNR_measure(img, gauss)
        a[k] = gauss.astype(np.uint8)
        # _, p[k] = make_profile(gauss.astype(np.uint8), th0=180.)
        z[k] = i
        k+=1
    t2 = time()
    print("Operation took : "+str((t2-t1))+" s.")
    print("Salt and Pepper")
    t1 = time()
    for val in range_salt_pepper:
        try:
            sp_noise = salt_pepper_noise(img,amount=val,s_vs_p = np.random.random_sample()+0.001)
        except ValueError:
            sp_noise = salt_pepper_noise(img, amount=val, s_vs_p=np.random.random_sample() + 0.001)
        snr[k] = SNR_measure(img, sp_noise)
        a[k] = sp_noise.astype(np.uint8)
        # _, p[k] = make_profile(sp_noise.astype(np.uint8), th0=180.)
        z[k] = i
        k += 1
    t2 = time()
    print("Operation took : " + str((t2 - t1)) + " s.")
    print("Poisson")
    t1 = time()
    for val in range_poisson_noise:
        try:
            p_noise = poisson_noise(img, offset= val )
        except ValueError:
            p_noise = poisson_noise(img, offset=val)
        snr[k] = SNR_measure(img, p_noise)
        a[k] = p_noise.astype(np.uint8)
        # _, p[k] = make_profile(p_noise.astype(np.uint8), th0=180.)
        z[k] = i
        k += 1
    t2 = time()
    print("Operation took : " + str((t2 - t1)) + " s.")
    print("Speckle")
    t1 = time()
    for val in range_speckle_noise:
        try:
            s_noise = speckle_noise(img, kernel=val)
        except ValueError:
            s_noise = speckle_noise(img, kernel=val)
        snr[k] = SNR_measure(img, s_noise)
        a[k] = s_noise.astype(np.uint8)
        # _, p[k] = make_profile(s_noise.astype(np.uint8), th0=180.)
        z[k] = i
        k += 1
    t2 = time()
    print("Operation took : " + str((t2 - t1)) + " s.")
print(k)
print(a.shape[0])
print(p.shape[0])
np.save(output_path_images,a)
np.save(output_path_snr,snr)
np.save(output_path_z,z)


# gauss = gaussian_noise(img)
# salt_pepper_noise = salt_pepper_noise(img)
# poisson_noise = poisson_noise(img)
# speckle_noise = speckle_noise(img)
# # cv2.imshow('image',img.astype(np.uint8))
# # cv2.imshow('gaussian_noise',gauss.astype(np.uint8))
# # cv2.imshow('salt_pepper_noise',salt_pepper_noise.astype(np.uint8))
# cv2.imshow('poisson_noise',poisson_noise.astype(np.uint8))
# # cv2.imshow('speckle_noise',speckle_noise.astype(np.uint8))
# cv2.waitKey(0)
