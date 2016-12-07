# -*- coding: utf-8 -*-
"""
    聚类和EM算法
    ~~~~~~~~~~~~~~~~

    聚类

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
# from  .agglomerative_clustering import test_AgglomerativeClustering,test_AgglomerativeClustering_nclusters,test_AgglomerativeClustering_linkage
# from .dbscan import test_DBSCAN,test_DBSCAN_epsilon,test_DBSCAN_min_samples
from chapters.Cluster_EM.gmm import test_GMM,test_GMM_cov_type,test_GMM_n_components
# from .kmeans import test_Kmeans,test_Kmeans_n_init,test_Kmeans_nclusters

def create_data(centers,num=100,std=0.7):
    '''
    生成用于聚类的数据集

    :param centers: 聚类的中心点组成的数组。如果中心点是二维的，则产生的每个样本都是二维的。
    :param num: 样本数
    :param std: 每个簇中样本的标准差
    :return: 用于聚类的数据集。是一个元组，第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return  X,labels_true
def plot_data(*data):
    '''
    绘制用于聚类的数据集

    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbyckm' # 每个簇的样本标记不同的颜色
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label,
		color=colors[i%len(colors)])

    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()

if __name__=='__main__':
    centers=[[1,1],[2,2],[1,2],[10,20]] # 用于产生聚类的中心点
    X,labels_true=create_data(centers,1000,0.5) # 产生用于聚类的数据集
    # plot_data(X,labels_true) # 绘制用于聚类的数据集
    # test_Kmeans(X,labels_true) #  调用 test_Kmeans 函数
    # test_Kmeans_nclusters(X,labels_true) #  调用 test_Kmeans_nclusters 函数
    # test_Kmeans_n_init(X,labels_true) #  调用 test_Kmeans_n_init 函数
    # test_DBSCAN(X,labels_true) #  调用 test_DBSCAN 函数
    # test_DBSCAN_epsilon(X,labels_true) #  调用 test_DBSCAN_epsilon 函数
    # test_DBSCAN_min_samples(X,labels_true) #  调用 test_DBSCAN_min_samples 函数
    # test_AgglomerativeClustering(X,labels_true) #  调用 test_AgglomerativeClustering 函数
    # test_AgglomerativeClustering_nclusters(X,labels_true) #  调用 test_AgglomerativeClustering_nclusters 函数
    # test_AgglomerativeClustering_linkage(X,labels_true) #  调用 test_AgglomerativeClustering_linkage 函数
    # test_GMM(X,labels_true) #  调用 test_GMM 函数
    # test_GMM_n_components(X,labels_true) #  调用 test_GMM_n_components 函数
    test_GMM_cov_type(X,labels_true) #  调用 test_GMM_cov_type 函数

