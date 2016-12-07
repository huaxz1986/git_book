# -*- coding: utf-8 -*-
"""
    决策树
    ~~~~~~~~~~~~~~~~

    DecisionTreeClassifier

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
from sklearn import cross_validation
import matplotlib.pyplot as plt
def load_data():
    '''
    加载用于分类问题的数据集。数据集采用 scikit-learn 自带的 iris 数据集

    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    iris=datasets.load_iris() # scikit-learn 自带的 iris 数据集
    X_train=iris.data
    y_train=iris.target
    return cross_validation.train_test_split(X_train, y_train,test_size=0.25,
		random_state=0,stratify=y_train)# 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
def test_DecisionTreeClassifier(*data):
    '''
    测试 DecisionTreeClassifier 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    print("Training score:%f"%(clf.score(X_train,y_train)))
    print("Testing score:%f"%(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_criterion(*data):
    '''
    测试 DecisionTreeClassifier 的预测性能随 criterion 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    criterions=['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print("criterion:%s"%criterion)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_splitter(*data):
    '''
    测试 DecisionTreeClassifier 的预测性能随划分类型的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    splitters=['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train, y_train)
        print("splitter:%s"%splitter)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_depth(*data,maxdepth):
    '''
    测试 DecisionTreeClassifier 的预测性能随 max_depth 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :param maxdepth: 一个整数，用于 DecisionTreeClassifier 的 max_depth 参数
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_scores=[]
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="traing score",marker='o')
    ax.plot(depths,testing_scores,label="testing score",marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data() # 产生用于分类问题的数据集
    test_DecisionTreeClassifier(X_train,X_test,y_train,y_test) # 调用 test_DecisionTreeClassifier
    # test_DecisionTreeClassifier_criterion(X_train,X_test,y_train,y_test) # 调用 test_DecisionTreeClassifier_criterion
    # test_DecisionTreeClassifier_splitter(X_train,X_test,y_train,y_test) # 调用 test_DecisionTreeClassifier_splitter
    # test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth=100) # 调用 test_DecisionTreeClassifier_depth
