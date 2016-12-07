# -*- coding: utf-8 -*-
"""
    贝叶斯分类器和贝叶斯网络
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    GaussianNB

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn import naive_bayes

def test_GaussianNB(*data):
    '''
    测试 GaussianNB 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    cls=naive_bayes.GaussianNB()
    cls.fit(X_train,y_train)
    print('Training Score: %.2f' % cls.score(X_train,y_train))
    print('Testing Score: %.2f' % cls.score(X_test, y_test))