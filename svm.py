# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(y)

X = iris.data[:,:2]
y = iris.target
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b', marker='*')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='g', marker='+')
plt.title('the relationship between sepal and target classes')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
# the second part: the classification which is based on Petal length & Petal width
# split the original dataset
# 0.7 of the dataset is used for train
X_train, X_test, y_train, y_test = train_test_split(iris.data[:, :2], iris.target, test_size=0.3, random_state=0)
# .fit() method is used to train SVM
lin_svc = svm.SVC(kernel='linear').fit(X_train, y_train)#线性核函数
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)#多项式核函数, 最高3阶
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)#高斯核函数

# the step of the grid
h = .02
# to create the grid plot to show data
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

# to plot the edge of different classes
for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
    # to create a 2*2 grid , and set the i image as current image
    plt.subplot(2, 2, i + 1)
    # to set the sub-plots in one plot
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # to predict, needs a  test
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # to plot the result
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()

# to calculate accuracy using the test set
lin_svc_pre = lin_svc.predict(X_test)
rbf_svc_pre = rbf_svc.predict(X_test)
poly_svc_pre = poly_svc.predict(X_test)
print('the acc of linear SVM = ', accuracy_score(y_test, lin_svc_pre))
print('the acc of RBF kernel SVM = ', accuracy_score(y_test, rbf_svc_pre))
print('the acc of polymonial SVM = ', accuracy_score(y_test, poly_svc_pre))
# the first part ends


# the second part: the classification which is based on Petal length & Petal width
# show the original data first
X = iris.data[:,2:]
y = iris.target
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b', marker='*')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='g', marker='+')
plt.title('the relationship between Petal and target classes')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()
# split dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data[:,2:], iris.target, test_size=0.3, random_state=0)
svc = svm.SVC(kernel='linear').fit(X_train, y_train)
lin_svc = svm.SVC(kernel='linear').fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)

h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ['LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
for i, clf in enumerate(( lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()

# calculating accuracy
lin_svc_pre = lin_svc.predict(X_test)
rbf_svc_pre = rbf_svc.predict(X_test)
poly_svc_pre = poly_svc.predict(X_test)
print('the acc of linear SVM = ', accuracy_score(y_test, lin_svc_pre))
print('the acc of RBF kernel SVM = ', accuracy_score(y_test, rbf_svc_pre))
print('the acc of polymonial SVM = ', accuracy_score(y_test, poly_svc_pre))