# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    嵌入式特征选择

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn.feature_selection import  SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_digits,load_diabetes
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def test_SelectFromModel():
    '''
    测试 SelectFromModel 的用法。

    :return: None
    '''
    digits=load_digits()
    X=digits.data
    y=digits.target
    estimator=LinearSVC(penalty='l1',dual=False)
    selector=SelectFromModel(estimator=estimator,threshold='mean')
    selector.fit(X,y)
    selector.transform(X)
    print("Threshold %s"%selector.threshold_)
    print("Support is %s"%selector.get_support(indices=True))
def test_Lasso(*data):
    '''
    测试 alpha 与稀疏性的关系

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
    X,y=data
    alphas=np.logspace(-2,2)
    zeros=[]
    for alpha in alphas:
        regr=Lasso(alpha=alpha)
        regr.fit(X,y)
        ### 计算零的个数 ###
        num=0
        for ele in regr.coef_:
            if abs(ele) < 1e-5:num+=1
        zeros.append(num)
    ##### 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,zeros)
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_ylim(0,X.shape[1]+1)
    ax.set_ylabel("zeros in coef")
    ax.set_title("Sparsity In Lasso")
    plt.show()
def test_LinearSVC(*data):
    '''
    测试 C  与 稀疏性的关系

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X,y=data
    Cs=np.logspace(-2,2)
    zeros=[]
    for C in Cs:
        clf=LinearSVC(C=C,penalty='l1',dual=False)
        clf.fit(X,y)
         ### 计算零的个数 ###
        num=0
        for row in clf.coef_:
            for ele in row:
                if abs(ele) < 1e-5:num+=1
        zeros.append(num)
    ##### 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,zeros)
    ax.set_xlabel("C")
    ax.set_xscale("log")
    ax.set_ylabel("zeros in coef")
    ax.set_title("Sparsity In SVM")
    plt.show()
if __name__=='__main__':
    test_SelectFromModel() # 调用 test_SelectFromModel
    # data=load_diabetes() # 生成用于回归问题的数据集
    # test_Lasso(data.data,data.target) # 调用 test_Lasso
    # data=load_digits() # 生成用于分类问题的数据集
    # test_LinearSVC(data.data,data.target) # 调用 test_LinearSVC