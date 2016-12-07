# -*- coding: utf-8 -*-
"""
    kNN和降维
    ~~~~~~~~~~

    Isomap

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import   datasets,manifold

def load_data():
    '''
    加载用于降维的数据

    :return: 一个元组，依次为训练样本集和样本集的标记
    '''
    iris=datasets.load_iris()# 使用 scikit-learn 自带的 iris 数据集
    return  iris.data,iris.target

def test_Isomap(*data):
    '''
    测试 Isomap 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、训练样本的标记
    :return: None
    '''
    X,y=data
    for n in [4,3,2,1]: # 依次考察降维目标为 4维、3维、2维、1维
        isomap=manifold.Isomap(n_components=n)
        isomap.fit(X)
        print('reconstruction_error(n_components=%d) : %s'%
            (n, isomap.reconstruction_error()))
def plot_Isomap_k(*data):
    '''
    测试 Isomap 中 n_neighbors 参数的影响，其中降维至 2维

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、训练样本的标记
    :return: None
    '''
    X,y=data
    Ks=[1,5,25,y.size-1] # n_neighbors参数的候选值的集合

    fig=plt.figure()
    for i, k in enumerate(Ks):
        isomap=manifold.Isomap(n_components=2,n_neighbors=k)
        X_r=isomap.fit_transform(X)#原始数据集转换到二维

        ax=fig.add_subplot(2,2,i+1)## 两行两列，每个单元显示不同 n_neighbors 参数的 Isomap 的效果图
        colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
            (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)# 颜色集合，不同标记的样本染不同的颜色
        for label ,color in zip( np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position,0],X_r[position,1],label="target= %d"
            %label,color=color)

        ax.set_xlabel("X[0]")
        ax.set_ylabel("X[1]")
        ax.legend(loc="best")
        ax.set_title("k=%d"%k)
    plt.suptitle("Isomap")
    plt.show()
def plot_Isomap_k_d1(*data):
    '''
    测试 Isomap 中 n_neighbors 参数的影响，其中降维至 1维

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、训练样本的标记
    :return: None
    '''
    X,y=data
    Ks=[1,5,25,y.size-1]# n_neighbors参数的候选值的集合

    fig=plt.figure()
    for i, k in enumerate(Ks):
        isomap=manifold.Isomap(n_components=1,n_neighbors=k)
        X_r=isomap.fit_transform(X)#原始数据集转换到 1 维

        ax=fig.add_subplot(2,2,i+1)## 两行两列，每个单元显示不同 n_neighbors 参数的 Isomap 的效果图
        colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
            (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)# 颜色集合，不同标记的样本染不同的颜色
        for label ,color in zip( np.unique(y),colors):
            position=y==label
            ax.scatter(X_r[position],np.zeros_like(X_r[position]),
            label="target= %d"%label,color=color)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="best")
        ax.set_title("k=%d"%k)
    plt.suptitle("Isomap")
    plt.show()
if __name__=='__main__':
    X,y=load_data() # 产生用于降维的数据集
    test_Isomap(X,y)   # 调用 test_Isomap
    #plot_Isomap_k(X,y)   # 调用 plot_Isomap_k
    #plot_Isomap_k_d1(X,y)   # 调用 plot_Isomap_k_d1
