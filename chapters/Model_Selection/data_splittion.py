# -*- coding: utf-8 -*-
"""
    模型选择
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    数据集切分

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,LeaveOneOut\
            ,cross_val_score
import  numpy as np
def test_train_test_split():
    '''
    测试  train_test_split 的用法

    :return:  None
    '''
    X=[[1,2,3,4],
       [11,12,13,14],
       [21,22,23,24],
       [31,32,33,34],
       [41,42,43,44],
       [51,52,53,54],
       [61,62,63,64],
       [71,72,73,74]]
    y=[1,1,0,0,1,1,0,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=0) # 切分，测试集大小为原始数据集大小的 40%
    print("X_train=",X_train)
    print("X_test=",X_test)
    print("y_train=",y_train)
    print("y_test=",y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,
             random_state=0,stratify=y) # 分层采样切分，测试集大小为原始数据集大小的 40%
    print("Stratify:X_train=",X_train)
    print("Stratify:X_test=",X_test)
    print("Stratify:y_train=",y_train)
    print("Stratify:y_test=",y_test)
def test_KFold():
    '''
    测试  KFold 的用法

    :return: None
    '''
    X=np.array([[1,2,3,4],
       [11,12,13,14],
       [21,22,23,24],
       [31,32,33,34],
       [41,42,43,44],
       [51,52,53,54],
       [61,62,63,64],
       [71,72,73,74],
       [81,82,83,84]])
    y=np.array([1,1,0,0,1,1,0,0,1])

    folder=KFold(n_splits=3,random_state=0,shuffle=False) # 切分之前不混洗数据集
    for train_index,test_index in folder.split(X,y):
          print("Train Index:",train_index)
          print("Test Index:",test_index)
          print("X_train:",X[train_index])
          print("X_test:",X[test_index])
          print("")

    shuffle_folder=KFold(n_splits=3,random_state=0,shuffle=True) # 切分之前混洗数据集
    for train_index,test_index in shuffle_folder.split(X,y):
          print("Shuffled Train Index:",train_index)
          print("Shuffled Test Index:",test_index)
          print("Shuffled X_train:",X[train_index])
          print("Shuffled X_test:",X[test_index])
          print("")
def test_StratifiedKFold():
    '''
    测试  StratifiedKFold 的用法

    :return: None
    '''
    X=np.array([[1,2,3,4],
       [11,12,13,14],
       [21,22,23,24],
       [31,32,33,34],
       [41,42,43,44],
       [51,52,53,54],
       [61,62,63,64],
       [71,72,73,74]])

    y=np.array([1,1,0,0,1,1,0,0])

    folder=KFold(n_splits=4,random_state=0,shuffle=False)
    stratified_folder=StratifiedKFold(n_splits=4,random_state=0,shuffle=False)
    for train_index,test_index in folder.split(X,y):
          print("Train Index:",train_index)
          print("Test Index:",test_index)
          print("y_train:",y[train_index])
          print("y_test:",y[test_index])
          print("")

    for train_index,test_index in stratified_folder.split(X,y):
          print("Stratified Train Index:",train_index)
          print("Stratified Test Index:",test_index)
          print("Stratified y_train:",y[train_index])
          print("Stratified y_test:",y[test_index])
          print("")
def test_LeaveOneOut():
    '''
    测试  LeaveOneOut 的用法

    :return: None
    '''
    X=np.array([[1,2,3,4],
       [11,12,13,14],
       [21,22,23,24],
       [31,32,33,34]]
    )
    y=np.array([1,1,0,0])

    lo=LeaveOneOut()
    for train_index,test_index in lo.split(X):
          print("Train Index:",train_index)
          print("Test Index:",test_index)
          print("X_train:",X[train_index])
          print("X_test:",X[test_index])
          print("")
def test_cross_val_score():
    '''
    测试  cross_val_score 的用法

    :return: None
    '''
    from sklearn.datasets import  load_digits
    from sklearn.svm import  LinearSVC

    digits=load_digits() # 加载用于分类问题的数据集
    X=digits.data
    y=digits.target

    result=cross_val_score(LinearSVC(),X,y,cv=10) # 使用 LinearSVC 作为分类器
    print("Cross Val Score is:",result)


if __name__=='__main__':
    # test_train_test_split() # 调用 test_train_test_split
    # test_KFold()# 调用 test_KFold
    test_StratifiedKFold()# 调用 test_StratifiedKFold
    # test_LeaveOneOut()# 调用 test_LeaveOneOut
    # test_cross_val_score()# 调用 test_cross_val_score
