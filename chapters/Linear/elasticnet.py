# -*- coding: utf-8 -*-
"""
    广义线性模型
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    ElasticNet

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,cross_validation

def load_data():
    '''
    加载用于回归问题的数据集

    :return: 一个元组，用于回归问题。元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
    '''
    diabetes = datasets.load_diabetes()#使用 scikit-learn 自带的一个糖尿病病人的数据集
    return cross_validation.train_test_split(datasets.data,diabetes.target,
		test_size=0.25,random_state=0) # 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

def test_ElasticNet(*data):
    '''
    测试 ElasticNet 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))
def test_ElasticNet_alpha_rho(*data):
    '''
    测试 ElasticNet 的预测性能随 alpha 和 l1_ratio 的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    alphas=np.logspace(-2,2)
    rhos=np.linspace(0.01,1)
    scores=[]
    for alpha in alphas:
            for rho in rhos:
                regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)
                regr.fit(X_train, y_train)
                scores.append(regr.score(X_test, y_test))
    ## 绘图
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig=plt.figure()
    ax=Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()
if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data() # 产生用于回归问题的数据集
    test_ElasticNet(X_train,X_test,y_train,y_test) # 调用 test_ElasticNet
    # test_ElasticNet_alpha_rho(X_train,X_test,y_train,y_test) # 调用 test_ElasticNet_alpha_rho