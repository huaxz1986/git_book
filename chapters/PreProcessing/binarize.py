# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    二元化

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn.preprocessing import Binarizer
def test_Binarizer():
    '''
    测试 Binarizer 的用法

    :return: None
    '''
    X=[   [1,2,3,4,5],
          [5,4,3,2,1],
          [3,3,3,3,3,],
          [1,1,1,1,1] ]
    print("before transform:",X)
    binarizer=Binarizer(threshold=2.5)
    print("after transform:",binarizer.transform(X))

if __name__=='__main__':
    test_Binarizer() # 调用 test_Binarizer