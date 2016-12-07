# -*- coding: utf-8 -*-
"""
    贝叶斯分类器和贝叶斯网络
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    贝叶斯分类器

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn import datasets,cross_validation,naive_bayes
import  matplotlib.pyplot as plt
from .gaussianNB import test_GaussianNB
from .multinomialNB import test_MultinomialNB,test_MultinomialNB_alpha
from .bernoulliNB import test_BernoulliNB,test_BernoulliNB_alpha,test_BernoulliNB_binarize
def load_data():
    '''
    加载用于分类问题的数据集。这里使用 scikit-learn 自带的 digits 数据集

    :return: 一个元组，用于分类问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的标记、测试样本集对应的标记
    '''
    digits=datasets.load_digits() # 加载 scikit-learn 自带的 digits 数据集
    return cross_validation.train_test_split(digits.data,digits.target,
		test_size=0.25,random_state=0,stratify=digits.target) #分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
def show_digits():
    '''
    绘制 digits 数据集。这里只是绘制数据集中前 25 个样本的图片。

    :return: None
    '''
    digits=datasets.load_digits()
    fig=plt.figure()
    print("vector from images 0:",digits.data[0])
    for i in range(25):
        ax=fig.add_subplot(5,5,i+1)
        ax.imshow(digits.images[i],cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data() # 产生用于分类问题的数据集
    test_GaussianNB(X_train,X_test,y_train,y_test) # 调用 test_GaussianNB
    test_MultinomialNB(X_train,X_test,y_train,y_test) # 调用 test_MultinomialNB
    test_MultinomialNB_alpha(X_train,X_test,y_train,y_test) # 调用 test_MultinomialNB_alpha
    test_BernoulliNB(X_train,X_test,y_train,y_test) # 调用 test_BernoulliNB
    test_BernoulliNB_alpha(X_train,X_test,y_train,y_test) # 调用 test_BernoulliNB_alpha
    test_BernoulliNB_binarize(X_train,X_test,y_train,y_test) # 调用 test_BernoulliNB_binarize
