# -*- coding: utf-8 -*-
"""
    感知机和神经网络
    ~~~~~~~~~~~~~~~~~

    神经网络模型：用于 iris 模型。注意 MLPClassifier 是 scikit-learn version 0.18 版本才出现的。截止目前，该版本还是开发版，官方提供的稳定版为 0.17。
    所以为了运行本示例，需要手动下载编译安装 scikit-learn version 0.18 版

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import numpy as np
from matplotlib import  pyplot as plt
from sklearn.neural_network import MLPClassifier

## 加载数据集
np.random.seed(0)
iris=datasets.load_iris() # 使用 scikit-learn  自带的 iris 数据集
X=iris.data[:,0:2] # 使用前两个特征，方便绘图
Y=iris.target # 标记值
data=np.hstack((X,Y.reshape(Y.size,1)))
np.random.shuffle(data) # 混洗数据。因为默认的iris 数据集：前50个数据是类别0，中间50个数据是类别1，末尾50个数据是类别2.混洗将打乱这个顺序
X=data[:,:-1]
Y=data[:,-1]
train_x=X[:-30]
test_x=X[-30:] # 最后30个样本作为测试集
train_y=Y[:-30]
test_y=Y[-30:]

def plot_classifier_predict_meshgrid(ax,clf,x_min,x_max,y_min,y_max):
      '''
     绘制 MLPClassifier 的分类结果

    :param ax:  Axes 实例，用于绘图
    :param clf: MLPClassifier 实例
    :param x_min: 第一维特征的最小值
    :param x_max: 第一维特征的最大值
    :param y_min: 第二维特征的最小值
    :param y_max: 第二维特征的最大值
    :return: None
      '''
      plot_step = 0.02 # 步长
      xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
          np.arange(y_min, y_max, plot_step))
      Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      ax.contourf(xx, yy, Z, cmap=plt.cm.Paired) # 绘图

def plot_samples(ax,x,y):
      '''
        绘制二维数据集

        :param ax:  Axes 实例，用于绘图
        :param x: 第一维特征
        :param y: 第二维特征
        :return: None
      '''
      n_classes = 3
      plot_colors = "bry" # 颜色数组。每个类别的样本使用一种颜色
      for i, color in zip(range(n_classes), plot_colors):
          idx = np.where(y == i)
          ax.scatter(x[idx, 0], x[idx, 1], c=color,
              label=iris.target_names[i], cmap=plt.cm.Paired) # 绘图

def mlpclassifier_iris():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集

        :return: None
        '''
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        classifier=MLPClassifier(activation='logistic',max_iter=10000,
            hidden_layer_sizes=(30,))
        classifier.fit(train_x,train_y)
        train_score=classifier.score(train_x,train_y)
        test_score=classifier.score(test_x,test_y)
        x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
        y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
        plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
        plot_samples(ax,train_x,train_y)
        ax.legend(loc='best')
        ax.set_xlabel(iris.feature_names[0])
        ax.set_ylabel(iris.feature_names[1])
        ax.set_title("train score:%f;test score:%f"%(train_score,test_score))
        plt.show()
def mlpclassifier_iris_hidden_layer_sizes():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 hidden_layer_sizes 的影响

        :return: None
        '''
        fig=plt.figure()
        hidden_layer_sizes=[(10,),(30,),(100,),(5,5),(10,10),(30,30)] # 候选的 hidden_layer_sizes 参数值组成的数组
        for itx,size in enumerate(hidden_layer_sizes):
            ax=fig.add_subplot(2,3,itx+1)
            classifier=MLPClassifier(activation='logistic',max_iter=10000
                ,hidden_layer_sizes=size)
            classifier.fit(train_x,train_y)
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("layer_size:%s;train score:%f;test score:%f"
                %(size,train_score,test_score))
        plt.show()
def mlpclassifier_iris_ativations():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 activation 的影响

        :return:  None
        '''
        fig=plt.figure()
        ativations=["logistic","tanh","relu"] # 候选的激活函数字符串组成的列表
        for itx,act in enumerate(ativations):
            ax=fig.add_subplot(1,3,itx+1)
            classifier=MLPClassifier(activation=act,max_iter=10000,
                hidden_layer_sizes=(30,))
            classifier.fit(train_x,train_y)
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("activation:%s;train score:%f;test score:%f"
                %(act,train_score,test_score))
        plt.show()
def mlpclassifier_iris_algorithms():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的 algorithm 的影响

        :return: None
        '''
        fig=plt.figure()
        algorithms=["l-bfgs","sgd","adam"] # 候选的算法字符串组成的列表
        for itx,algo in enumerate(algorithms):
            ax=fig.add_subplot(1,3,itx+1)
            classifier=MLPClassifier(activation="tanh",max_iter=10000,
                hidden_layer_sizes=(30,),algorithm=algo)
            classifier.fit(train_x,train_y)
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("algorithm:%s;train score:%f;test score:%f"%(algo,train_score,test_score))
        plt.show()

def mlpclassifier_iris_eta():
        '''
        使用 MLPClassifier 预测调整后的 iris 数据集。考察不同的学习率的影响

        :return: None
        '''
        fig=plt.figure()
        etas=[0.1,0.01,0.001,0.0001] # 候选的学习率值组成的列表
        for itx,eta in enumerate(etas):
            ax=fig.add_subplot(2,2,itx+1)
            classifier=MLPClassifier(activation="tanh",max_iter=1000000,
            hidden_layer_sizes=(30,),algorithm='sgd',learning_rate_init=eta)
            classifier.fit(train_x,train_y)
            iter_num=classifier.n_iter_
            train_score=classifier.score(train_x,train_y)
            test_score=classifier.score(test_x,test_y)
            x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 2
            y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 2
            plot_classifier_predict_meshgrid(ax,classifier,x_min,x_max,y_min,y_max)
            plot_samples(ax,train_x,train_y)
            ax.legend(loc='best')
            ax.set_xlabel(iris.feature_names[0])
            ax.set_ylabel(iris.feature_names[1])
            ax.set_title("eta:%f;train score:%f;test score:%f;iter_num:%d"
                %(eta,train_score,test_score,iter_num))
        plt.show()

if __name__=='__main__':
    mlpclassifier_iris()     # 调用 mlpclassifier_iris
    #mlpclassifier_iris_hidden_layer_sizes()# 调用 mlpclassifier_iris_hidden_layer_sizes
    #mlpclassifier_iris_ativations()# 调用 mlpclassifier_iris_ativations
    #mlpclassifier_iris_algorithms()# 调用 mlpclassifier_iris_algorithms
    #mlpclassifier_iris_eta()# 调用 mlpclassifier_iris_eta
