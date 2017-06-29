# -*- coding: utf-8 -*-
"""
    支持向量机
    ~~~~~~~~~~~~~~~~

    LinearSVC

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,cross_validation,svm

def load_data_classfication():
    '''
    加载用于分类问题的数据集

    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    iris=datasets.load_iris() # 使用 scikit-learn 自带的 iris 数据集
    X_train=iris.data
    y_train=iris.target
    return cross_validation.train_test_split(X_train, y_train,test_size=0.25,
		random_state=0,stratify=y_train) # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

def test_LinearSVC(*data):
    '''
    测试 LinearSVC 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    cls=svm.LinearSVC()
    cls.fit(X_train,y_train)
    print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
    print('Score: %.2f' % cls.score(X_test, y_test))
def test_LinearSVC_loss(*data):
    '''
    测试 LinearSVC 的预测性能随损失函数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    losses=['hinge','squared_hinge']
    for loss in losses:
        cls=svm.LinearSVC(loss=loss)
        cls.fit(X_train,y_train)
        print("Loss:%s"%loss)
        print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
        print('Score: %.2f' % cls.score(X_test, y_test))
def test_LinearSVC_L12(*data):
    '''
    测试 LinearSVC 的预测性能随正则化形式的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    L12=['l1','l2']
    for p in L12:
        cls=svm.LinearSVC(penalty=p,dual=False)
        cls.fit(X_train,y_train)
        print("penalty:%s"%p)
        print('Coefficients:%s, intercept %s'%(cls.coef_,cls.intercept_))
        print('Score: %.2f' % cls.score(X_test, y_test))
def test_LinearSVC_C(*data):
    '''
    测试 LinearSVC 的预测性能随参数 C 的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:   None
    '''
    X_train,X_test,y_train,y_test=data
    Cs=np.logspace(-2,1)
    train_scores=[]
    test_scores=[]
    for C in Cs:
        cls=svm.LinearSVC(C=C)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(Cs,train_scores,label="Traing score")
    ax.plot(Cs,test_scores,label="Testing score")
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("LinearSVC")
    ax.legend(loc='best')
    plt.show()
if __name__=="__main__":
    X_train,X_test,y_train,y_test=load_data_classfication() # 生成用于分类的数据集
    test_LinearSVC(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC
    # test_LinearSVC_loss(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC_loss
    # test_LinearSVC_L12(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC_L12
    # test_LinearSVC_C(X_train,X_test,y_train,y_test) # 调用 test_LinearSVC_C
