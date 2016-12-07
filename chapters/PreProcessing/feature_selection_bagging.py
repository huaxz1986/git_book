# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    包裹式特征选择

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn.feature_selection import  RFE,RFECV
from sklearn.svm import LinearSVC
from sklearn.datasets import  load_iris
from  sklearn import  cross_validation

def test_RFE():
    '''
    测试 RFE 的用法，其中目标特征数量为 2

    :return: None
    '''
    iris=load_iris()
    X=iris.data
    y=iris.target
    estimator=LinearSVC()
    selector=RFE(estimator=estimator,n_features_to_select=2)
    selector.fit(X,y)
    print("N_features %s"%selector.n_features_)
    print("Support is %s"%selector.support_)
    print("Ranking %s"%selector.ranking_)
def test_RFECV():
    '''
    测试 RFECV 的用法

    :return:  None
    '''
    iris=load_iris()
    X=iris.data
    y=iris.target
    estimator=LinearSVC()
    selector=RFECV(estimator=estimator,cv=3)
    selector.fit(X,y)
    print("N_features %s"%selector.n_features_)
    print("Support is %s"%selector.support_)
    print("Ranking %s"%selector.ranking_)
    print("Grid Scores %s"%selector.grid_scores_)
def test_compare_with_no_feature_selection():
    '''
    比较经过特征选择和未经特征选择的数据集，对 LinearSVC 的预测性能的区别

    :return: None
    '''
    ### 加载数据
    iris=load_iris()
    X,y=iris.data,iris.target
    ### 特征提取
    estimator=LinearSVC()
    selector=RFE(estimator=estimator,n_features_to_select=2)
    X_t=selector.fit_transform(X,y)
    #### 切分测试集与验证集
    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X, y,
                test_size=0.25,random_state=0,stratify=y)
    X_train_t,X_test_t,y_train_t,y_test_t=cross_validation.train_test_split(X_t, y,
                test_size=0.25,random_state=0,stratify=y)
    ### 测试与验证
    clf=LinearSVC()
    clf_t=LinearSVC()
    clf.fit(X_train,y_train)
    clf_t.fit(X_train_t,y_train_t)
    print("Original DataSet: test score=%s"%(clf.score(X_test,y_test)))
    print("Selected DataSet: test score=%s"%(clf_t.score(X_test_t,y_test_t)))
if __name__=='__main__':
    test_RFE() # 调用 test_RFE
    test_compare_with_no_feature_selection() # 调用 test_compare_with_no_feature_selection
    test_RFECV() # 调用 test_RFECV