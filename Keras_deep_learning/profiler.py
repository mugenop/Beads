import numpy as np
import cv2

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib_scalebar.scalebar import ScaleBar
import lmfit
import itertools
import time

import os

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


def calculateProfil(parameters, x, y=None, fit=True):
    v = parameters.valuesdict()
    xx = (x - v['x0']) ** 2 / (2 * v['sigma'] ** 2)
    th_gray = v['A'] * np.exp(-(xx**v['k']))+ v['b']

    if fit:
        return (y - th_gray)
    else:
        return th_gray


# path_in = 'F:/Experiments/S/output/'
# path_in += 'profiles/'
# path_out = path_in+'Images/'
# dx = 1
# size = 1
# fl = 1
# listf = os.listdir(path_in)
# if not os.path.isdir(path_out):
#     os.mkdir(path_out)
# format = '.png'
# i = 0
# a_list = []
# k_list = []
# sigma_list = []
# x_exp_list = []
# y_exp_list = []
# x_th_list = []
# y_th_list = []
# leg = []
# for d in listf:
#     try:
#
#         str = d.split('.')
#         data = np.loadtxt(path_in+d,delimiter=',',skiprows=1).transpose()
#         X = (data[0]*dx)/size
#         Y = data[1]
#         Y = (-Y+255)/255.
#         p = lmfit.Parameters()
#         p.add('x0',value=1.0,min=-2*np.amin(X),max = 2*np.amax(X),)
#         p.add('A', value=1, min=0, max=2.,)
#         p.add('sigma', value=0.95, min=0.0000001, max=10.,)
#         p.add('k', value=1.01475011, min=0.0000001, max=100,)
#         p.add('b', value=0, min=-10, max=2*np.amin(Y),vary=False)
#         mi = lmfit.minimize(calculateProfil, p,
#                         args=(X, Y, True), method='leastsq',
#                         maxfev=1000000)
#         print (mi.success)
#         lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
#         plt.plot(X,Y,'ro',label= 'experimental')
#         x_th = np.arange(np.amin(X),np.amax(X),0.0001)
#         y_th = calculateProfil(mi.params,x_th,fit=False)
#         plt.plot(x_th,y_th,label = 'fit')
#         plt.title(''.join(str[:-1]))
#         plt.xlabel('profil radius/size of bead')
#         plt.ylabel('Level of gray/255')
#         plt.legend()
#         plt.savefig(path_out+''.join(str[:-1])+format)
#         plt.close()
#
#         v = mi.params.valuesdict()
#
#         a_list.append( v['A'])
#         k_list.append(v['k'])
#         sigma_list.append(v['sigma'])
#         x_exp_list.append(X)
#         x_th_list.append(x_th)
#         y_exp_list.append(Y)
#         y_th_list.append(y_th)
#         leg.append(''.join(str[:-1]))
#
#     except IOError:
#         continue
# Z = np.arange(0,130,10)/fl


# plt.plot(Z,np.array(a_list),'-o')
# plt.title('Evolution of amplitude as function of the height')
# plt.xlabel('Z (normalized by focal length)')
# plt.ylabel('A')
# plt.savefig(path_out + 'A(Z)'+ format)
# plt.close()
#
# plt.plot(Z,np.array(k_list),'-o')
# plt.title('Evolution of power index  as function of the height')
# plt.xlabel('Z (normalized by focal length)')
# plt.ylabel('K')
# plt.savefig(path_out + 'K(Z)' + format)
# plt.close()
#
# plt.plot(Z,np.log(np.array(k_list)),'-o')
# plt.title('Evolution of power index  as function of the height')
# plt.xlabel('Z (normalized by focal length)')
# plt.ylabel('Log (K)')
# plt.savefig(path_out + 'Log(K(Z))' + format)
# plt.close()
#
# plt.plot(Z,np.array(sigma_list),'-o')
# plt.title('Evolution of the standard deviation  as function of the height')
# plt.xlabel('Z (normalized by focal length)')
# plt.ylabel(r'$\sigma$')
# plt.savefig(path_out + 'sigma(Z)' + format)
# plt.close()
#
# marker = itertools.cycle(('.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_',))
# cycle = 0
# fig = plt.figure(figsize=(15, 15), dpi=300)
#
# for cycle in range(len(y_th_list)):
#     print(cycle)
#     plt.plot(x_exp_list[cycle],y_exp_list[cycle], marker = marker.next(), linestyle='-',label =leg[cycle] )
#
# plt.title('Experimental Profiles evolution')
# plt.xlabel('profil radius/size of bead')
# plt.ylabel('Level of gray Normalized')
# plt.legend()
# plt.savefig(path_out+'Experimental Profiles evolution'+format)
# plt.close()
# fig = plt.figure(figsize=(15, 15), dpi=300)
#
# for cycle in range(len(y_th_list)):
#     print(cycle)
#     plt.plot(x_th_list[cycle], y_th_list[cycle],label=leg[cycle])
#
# plt.title('Fit Profiles evolution')
# plt.xlabel('profil radius/size of bead')
# plt.ylabel('Level of gray Normalized')
# plt.legend()
# plt.savefig(path_out+'fitted Profiles evolution'+format)
# plt.close()


if __name__ == '__main__':
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'normal',
            'size': 30,
            }
    path_in = "G:/Keras/"
    path_out = "G:/Keras/Images/"
    dx = 1
    size = 1
    fl = 1
    listf = os.listdir(path_in)
    listf.sort()
    # for d in listf:
    #     img = cv2.imread(path_in + d, 0)
    #     d_img = d.split('.')
    #
    #     im = d_img[0].split('_')
    #     num = im[1]
    #
    #     if len(num)<2:
    #         num = '00'+num
    #     elif len(num)<3:
    #         num = '0' + num
    #     im[1] = num
    #     im = '_'.join(im)
    #     d_img[0] = im
    #     d = '.'.join(d_img)
    #     print (d)
    #     cv2.imwrite(path_out+d,img)
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    print(listf)
    X = np.load(path_in+'images_lessNoise.npy')
    print(X.shape)
    profiles = np.zeros(shape=(X.shape[0],int(X.shape[2]/2)), dtype=np.float)
    t1 = time.time()
    timeer = []
    for d in range(0,X.shape[0]):

        _,profiles[d] = make_profile(X[d],th0=180)
        # plt.plot(profiles[d])
        # plt.pause(1)
        # plt.close()
        shower = 100
        if d%shower == 0:

            print (d)
            timeer.append(time.time()-t1)
            try:
                print ("ETA: ", ((X.shape[0]-d)/(60.*shower))*np.mean(timeer[1:]))
            except:
                continue
            t1 = time.time()


    np.save(path_in+"profiles_lessNoise.npy", profiles)