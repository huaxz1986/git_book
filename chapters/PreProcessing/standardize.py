# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    数据标准化

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler

def test_MinMaxScaler():
    '''
    测试 MinMaxScaler 的用法

    :return: None
    '''
    X=[   [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4,],
      [4,8,7,8,1] ]
    print("before transform:",X)
    scaler=MinMaxScaler(feature_range=(0,2))
    scaler.fit(X)
    print("min_ is :",scaler.min_)
    print("scale_ is :",scaler.scale_)
    print("data_max_ is :",scaler.data_max_)
    print("data_min_ is :",scaler.data_min_)
    print("data_range_ is :",scaler.data_range_)
    print("after transform:",scaler.transform(X))
def test_MaxAbsScaler():
    '''
    测试 MaxAbsScaler 的用法

    :return: None
    '''
    X=[   [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4,],
      [4,8,7,8,1] ]
    print("before transform:",X)
    scaler=MaxAbsScaler()
    scaler.fit(X)
    print("scale_ is :",scaler.scale_)
    print("max_abs_ is :",scaler.max_abs_)
    print("after transform:",scaler.transform(X))
def test_StandardScaler():
    '''
    测试 StandardScaler 的用法

    :return: None
    '''
    X=[   [1,5,1,2,10],
      [2,6,3,2,7],
      [3,7,5,6,4,],
      [4,8,7,8,1] ]
    print("before transform:",X)
    scaler=StandardScaler()
    scaler.fit(X)
    print("scale_ is :",scaler.scale_)
    print("mean_ is :",scaler.mean_)
    print("var_ is :",scaler.var_)
    print("after transform:",scaler.transform(X))

if __name__=='__main__':
    test_MinMaxScaler()  # 调用 test_MinMaxScaler
    # test_MaxAbsScaler()  # 调用 test_MaxAbsScaler
    # test_MaxAbsScaler()  # 调用 test_MaxAbsScaler