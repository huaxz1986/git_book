# -*- coding: utf-8 -*-
"""
    模型选择
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    学习曲线

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.learning_curve import learning_curve

def test_learning_curve():
    '''
    测试 learning_curve 的用法 。验证对于 LinearSVC 分类器 ，数据集的大小对于预测性能的影响

    :return:
    '''
    ### 加载数据
    digits = load_digits()
    X,y=digits.data,digits.target
    #### 获取学习曲线 ######
    train_sizes=np.linspace(0.1,1.0,endpoint=True,dtype='float')
    abs_trains_sizes,train_scores, test_scores = learning_curve(LinearSVC(),
            X, y,cv=10, scoring="accuracy",train_sizes=train_sizes)
    ###### 对每个 C ，获取 10 折交叉上的预测得分上的均值和方差 #####
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ####### 绘图 ######
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    ax.plot(abs_trains_sizes, train_scores_mean, label="Training Accuracy", color="r")
    ax.fill_between(abs_trains_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    ax.plot(abs_trains_sizes, test_scores_mean, label="Testing Accuracy", color="g")
    ax.fill_between(abs_trains_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")

    ax.set_title("Learning Curve with LinearSVC")
    ax.set_xlabel("Sample Nums")
    ax.set_ylabel("Score")
    ax.set_ylim(0,1.1)
    ax.legend(loc='best')
    plt.show()

if __name__=="__main__":
    test_learning_curve() # 调用 test_learning_curve