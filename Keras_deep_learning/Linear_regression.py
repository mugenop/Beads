import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import pickle
# matplotlib.use("Agg")
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os

EPOCHS = 100
INIT_LR = 1e-2
BS = 32
path_images = "C:/Users/Mugen/PycharmProjects/Keras_deep_learning/Images/"
list_images = os.listdir(path_images)
path_in = "G:/Keras/"
path_model = path_in+"model_regression.sav"
path_save_fig = path_in+"model_output_model_regression"


def scorer(y_test,y_predict):

    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.4f"
          % mean_squared_error(y_test, y_predict))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.4f' % r2_score(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

X = np.load(path_in+"profiles_lessNoise.npy")
X = X.reshape((X.shape[0], -1))

Z_labels = np.load(path_in+"z_lessNoise.npy")
Z_base = np.load(path_in+"zs.npy").astype(np.float64)
l = int(Z_labels.shape[0]/Z_base.shape[0])
for i in range(len(list_images)):
    Z_labels[i*l:(i+1)*l] = Z_base[i]
index = np.where(Z_labels>=-0.001)
# Z_labels = Z_labels[index]
D = 10
Z_labels = Z_labels [::D]
# X = X[index]
X = X[::D]

n_iter = 10000
X_train, X_test, y_train, y_test = train_test_split(X, Z_labels, test_size=0.20, random_state=42)
# regr = linear_model.SGDRegressor(average=100, max_iter=10**4,verbose=1,early_stopping=True,validation_fraction=0.2,n_iter_no_change=100,power_t=0.1,alpha=0.001,shuffle=True)

# Score = regr.fit(X_train.astype(np.float32),y_train,)
#
#
# X_test = X_test
# Y_test = y_test
# y_pred = regr.predict(X_test.astype(np.float64))
# arr1inds = Y_test.argsort()
#
# y_test = Y_test[arr1inds[::-1]]
# y_pred = y_pred[arr1inds[::-1]]
# # print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(y_test, y_pred))
# # Explained variance score: 1 is perfect prediction
# print('R2 score: %.2f' % r2_score(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
#
# # while r2 < 0.9:
# #     X_1, X_2, Y_1, Y_2 = train_test_split(X, Z_labels, test_size=0.5)
# #     X_11, X_12, Y_11, Y_12 = train_test_split(X_1, Y_1, test_size=0.5)
# #     X_21, X_22, Y_21, Y_22 = train_test_split(X_2, Y_2, test_size=0.5)
# #     X_221, X_test, Y_221, Y_test = train_test_split(X_22, Y_22, test_size=0.25)
# #     x = [X_11, X_12, X_21, X_221]
# #     y = [Y_11, Y_12, Y_21, Y_221]
# #     for i in range(4):
# #         scores1 = regr.fit(x[i], y[i])
# #         print(scores1)
# #
# #     y_pred = regr.predict(X_test)
# #     arr1inds = Y_test.argsort()
# #
# #     y_test = Y_test[arr1inds[::-1]]
# #     y_pred = y_pred[arr1inds[::-1]]
# #     # print('Coefficients: \n', regr.coef_)
# #     # The mean squared error
# #     print("Mean squared error: %.2f"
# #           % mean_squared_error(y_test, y_pred))
# #     # Explained variance score: 1 is perfect prediction
# #     print('R2 score: %.2f' % r2_score(y_test, y_pred))
# #     r2 = r2_score(y_test, y_pred)
# #
#
# # Plot outputs
# plt.plot(y_pred, color='blue', linewidth=3,label = "Prediction")
# plt.plot(y_test, "r-o", linewidth=3, label= "Truth")
# plt.xticks(())
# plt.yticks(())
# plt.savefig(path_save_fig+".png")
# plt.show()
#
#
# pickle.dump(regr, open(path_model, 'wb'))



svr_rbf = SVR(kernel='rbf', C=100, gamma=100.,tol=1e-10, max_iter=100000,verbose=1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
# y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
# y_poly = svr_poly.fit(X_train, y_train).predict(X_test)

arr1inds = y_test.argsort()

y_test = y_test[arr1inds[::-1]]
y_rbf = y_rbf[arr1inds[::-1]]
# y_lin = y_lin[arr1inds[::-1]]
# y_poly = y_poly[arr1inds[::-1]]

# #############################################################################
# Look at the results
lw = 2
# X_train, X_test, y_train, y_test
scorer(y_test,y_rbf)
# scorer(y_test,y_lin)
# scorer(y_test,y_poly)
plt.plot(y_test, color='darkorange', label='data')
plt.plot(y_rbf, color='navy', lw=lw, label='RBF model')
# plt.plot(y_lin, color='c', lw=lw, label='Linear model')
# plt.plot(y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(path_save_fig+".png")
plt.show()
pickle.dump(svr_rbf, open(path_model, 'wb'))

