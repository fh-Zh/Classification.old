from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# same process as svm
np.random.seed(0)
iris = datasets.load_iris()
X = iris.data[:, 0:2]
y = iris.target
train_x, test_x, train_y, test_y = train_test_split(iris.data[:, :2], iris.target, test_size=0.3, random_state=0)

#hidden_layer_sizes=[(10,),(30,),(100,),(5,5),(10,10),(30,30)] #可选的神经元层数
#ativations=["logistic","tanh","relu"] #可选的激活函数
#learnrates=[0.1,0.01,0.001] #可选的学习率
solvers=["lbfgs","sgd","adam"] #可选的solver
for i, sol in enumerate(solvers):
    classifier = MLPClassifier(activation="tanh", max_iter=1000000,
                               hidden_layer_sizes=(10,5), solver=sol, learning_rate_init=0.01)
    classifier.fit(train_x, train_y)
    train_score = classifier.score(train_x, train_y)
    print('when solver =', sol, '\n','train_score=',train_score)
    test_score = classifier.score(test_x, test_y)
    print('test_score=',test_score,'\n')
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
    plot_step = 0.02  # 步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 3, i + 1)
    plt.subplots_adjust(wspace=0.3, hspace=1)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(sol)
plt.show()