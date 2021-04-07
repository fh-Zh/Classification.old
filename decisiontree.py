# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

# loading data
data = load_iris()
# get four features
feature_names = data.get('feature_names')
# get 'values' of features
x = data.get('data')
# get labels
y = data.get('target')

# sizeof data
num = x.shape[0]
# 0.7 for train while 0.3 for test
num_test = int(num * 0.3)
num_train = num - num_test
# random data
index = np.arange(num)
np.random.shuffle(index)
# split data
x_test = x[index[:num_test], :]
y_test = y[index[:num_test]]
x_train = x[index[num_test:], :]
y_train = y[index[num_test:]]

# declare and train decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

# to predict
y_test_pre = clf.predict(x_test)
print('the prediction values: ', y_test_pre)
print('the actual values: ', y_test)

# calculate acc
acc = sum(y_test_pre == y_test)/num_test
print('the acc of decision tree = ', acc, '=', sum(y_test_pre == y_test), '/', num_test)

# visualize the original data
f = []
f.append(y == 0)
f.append(y == 1)
f.append(y == 2)
# each label has a color
color = ['red', 'blue', 'green']
# four features can have a combination of
fig, axes = plt.subplots(4, 4)
for i, ax in enumerate(axes.flat):
    row = i // 4
    col = i % 4
    if row == col:
        ax.text(.1,.5, feature_names[row])
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    for k in range(3):
        ax.scatter(x[f[k], row], x[f[k], col], c=color[k], s=3)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()