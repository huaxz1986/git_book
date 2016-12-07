# -*- coding: utf-8 -*-
"""
    模型选择
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    回归问题性能度量

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn.metrics import mean_absolute_error,mean_squared_error

def test_mean_absolute_error():
    '''
    测试 mean_absolute_error 的用法

    :return: None
    '''
    y_true=[1,1,1,1,1,2,2,2,0,0]
    y_pred=[0,0,0,1,1,1,0,0,0,0]

    print("Mean Absolute Error:",mean_absolute_error(y_true,y_pred))
def test_mean_squared_error():
    '''
    测试 mean_squared_error 的用法

    :return: None
    '''
    y_true=[1,1,1,1,1,2,2,2,0,0]
    y_pred=[0,0,0,1,1,1,0,0,0,0]

    print("Mean Absolute Error:",mean_absolute_error(y_true,y_pred))
    print("Mean Square Error:",mean_squared_error(y_true,y_pred))

if __name__=="__main__":
    test_mean_absolute_error() # 调用  test_mean_absolute_error()
    # test_mean_squared_error() # 调用  test_mean_squared_error()