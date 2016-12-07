# -*- coding: utf-8 -*-
"""
    模型选择
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    损失函数

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn.metrics import zero_one_loss,log_loss


def test_zero_one_loss():
    '''
    测试 0-1 损失函数

    :return: None
    '''
    y_true=[1,1,1,1,1,0,0,0,0,0]
    y_pred=[0,0,0,1,1,1,1,1,0,0]
    print("zero_one_loss<fraction>:",zero_one_loss(y_true,y_pred,normalize=True))
    print("zero_one_loss<num>:",zero_one_loss(y_true,y_pred,normalize=False))
def test_log_loss():
    '''
    测试对数损失函数

    :return:  None
    '''
    y_true=[1, 1, 1, 0, 0, 0]
    y_pred=[[0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1]]
    print("log_loss<average>:",log_loss(y_true,y_pred,normalize=True))
    print("log_loss<total>:",log_loss(y_true,y_pred,normalize=False))

if __name__=="__main__":
    test_zero_one_loss() # 调用 test_zero_one_loss
    # test_log_loss() # 调用 test_log_loss