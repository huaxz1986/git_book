# -*- coding: utf-8 -*-
"""
    数据预处理
    ~~~~~~~~~~~~~~~~

    流水线

    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""

from sklearn.svm import  LinearSVC
from sklearn.datasets import  load_digits
from sklearn import  cross_validation
from sklearn.linear_model import LogisticRegression
from  sklearn.pipeline import Pipeline
def test_Pipeline(data):
    '''
    测试 Pipeline 的用法

    :param data:  一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    steps=[("Linear_SVM",LinearSVC(C=1,penalty='l1',dual=False)),
           ("LogisticRegression",LogisticRegression(C=1))]
    pipeline=Pipeline(steps)
    pipeline.fit(X_train,y_train)
    print("Named steps:",pipeline.named_steps)
    print("Pipeline Score:",pipeline.score(X_test,y_test))
if __name__=='__main__':
    data=load_digits() # 生成用于分类问题的数据集
    test_Pipeline(cross_validation.train_test_split(data.data, data.target,test_size=0.25
			,random_state=0,stratify=data.target)) # 调用 test_Pipeline
