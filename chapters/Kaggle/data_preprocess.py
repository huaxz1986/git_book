import  os
import  pickle
import numpy as np
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing  import OneHotEncoder,MaxAbsScaler
from data_clean import current_time
from data_clean import Data_Cleaner

class Data_Preprocesser:
    '''
    数据预处理器

    它的初始化需要提供清洗好的数据。它提供了唯一的对外接口：load_data()。它返回预处理好的数据。
    如果数据已存在，则直接返回。否则将执行一系列预处理操作并返回预处理好的数据。
    '''
    def __init__(self,train_datas,test_datas):
        '''
        :param train_datas: 清洗好的训练集
        :param test_datas: 清洗好的测试集
        :return:
        '''
        self.types=train_datas.keys()
        self.train_datas=train_datas
        self.test_datas=test_datas

        self.fname='output/processed_data'
    def load_data(self):
        '''
        加载预处理好的数据

        如果数据已经存在，则直接返回。如果不存在，预处理数据，并且存储之后返回。

        :return:一个元组：依次为：train_datas,test_datas
        '''
        if(self._is_ready()):
            print("preprocessed data is availiable!\n")
            self._load_data()
        else:
            self._onehot_encode()
            self._scaled()
            self._save_data()
        return self.train_datas,self.test_datas
    def _onehot_encode(self):
        '''
        独热码编码

        :return:
        '''
        print("----- Begin run onehot_encode at %s -------"%current_time())
        train_results={}
        test_results={}
        self.encoders={}

        for _type in self.types:
            if _type=='type 1':
                one_hot_cols=['char_%d_act'%i for i in range(1,10)]+['char_%d_people'%i for i in range(1,10)]
                train_end_cols=['group_1','date_act','date_people','char_38','outcome']
                test_end_cols=['group_1','date_act','date_people','char_38']
            else:
                one_hot_cols=['char_%d_people'%i for i in range(1,10)]
                train_end_cols=['group_1','char_10_act','date_act','date_people','char_38','outcome']
                test_end_cols=['group_1','char_10_act','date_act',	'date_people','char_38']

            train_front_array=self.train_datas[_type][one_hot_cols].values #头部数组
            train_end_array=self.train_datas[_type][train_end_cols].values#末尾数组
            train_middle_array=self.train_datas[_type].drop(train_end_cols+one_hot_cols,axis=1,inplace=False).values#中间数组

            test_front_array=self.test_datas[_type][one_hot_cols].values #头部数组
            test_end_array=self.test_datas[_type][test_end_cols].values#末尾数组
            test_middle_array=self.test_datas[_type].drop(test_end_cols+one_hot_cols,axis=1,inplace=False).values#中间数组

            encoder=OneHotEncoder(categorical_features='all',sparse=True)# 一个稀疏矩阵，类型为 csr_matrix
            train_result=hstack([encoder.fit_transform(train_front_array),csr_matrix(train_middle_array),csr_matrix(train_end_array)])
            test_result=hstack([encoder.transform(test_front_array),csr_matrix(test_middle_array),csr_matrix(test_end_array)])
            train_results[_type]=train_result
            test_results[_type]=test_result
            self.encoders[_type]=encoder

        self.train_datas=train_results
        self.test_datas=test_results
        print("----- End run onehot_encode at %s -------"%current_time())
    def _scaled(self):
        '''
        特征归一化，采用 MaxAbsScaler 来进行归一化
        :return:
        '''
        print("----- Begin run scaled at %s -------"%current_time())
        train_scales={}
        test_scales={}
        self.scalers={}
        for _type in self.types:
            if _type=='type 1':
                    train_last_index=5#最后5列为 group_1/date_act/date_people/char_38/outcome
                    test_last_index=4#最后4列为 group_1/date_act/date_people/char_38
            else:
                    train_last_index=6#最后6列为 group_1/char_10_act/date_act/date_people/char_38/outcome
                    test_last_index=5#最后5列为 group_1/char_10_act/date_act/date_people/char_38

            scaler=MaxAbsScaler()
            train_array=self.train_datas[_type].toarray()
            train_front=train_array[:,:-train_last_index]
            train_mid=scaler.fit_transform(train_array[:,-train_last_index:-1])#outcome 不需要归一化
            train_end=train_array[:,-1].reshape((-1,1)) #outcome
            train_scales[_type]=np.hstack((train_front,train_mid,train_end))

            test_array=self.test_datas[_type].toarray()
            test_front=test_array[:,:-test_last_index]
            test_end=scaler.transform(test_array[:,-test_last_index:])
            test_scales[_type]=np.hstack((test_front,test_end))
            self.scalers[_type]=scaler
        self.train_datas=train_scales
        self.test_datas=test_scales
        print("----- End run scaled at %s -------"%current_time())
    def _is_ready(self):
        if(os.path.exists(self.fname)):
            return True
        else :
            return False
    def _save_data(self):
        print("----- Begin run save_data at %s -------"%current_time())
        with open(self.fname,'wb') as file:#保存训练集、测试集、编码器、归一化器
            pickle.dump([self.train_datas,self.test_datas,self.encoders,self.scalers],file)
        print("----- End run save_data at %s -------"%current_time())
    def _load_data(self):
        print("----- Begin run _load_data at %s -------"%current_time())
        with open(self.fname,'rb') as file:#加载训练集、测试集、编码器、归一化器
            self.train_datas,self.test_datas,self.encoders,self.scalers=pickle.load(file)
        print("----- End run _load_data at %s -------"%current_time())
if __name__=='__main__':
    clearner=Data_Cleaner("./data/people.csv",'./data/act_train.csv','./data/act_test.csv')
    result=clearner.load_data()
    preprocessor=Data_Preprocesser(*result)
    result=preprocessor.load_data()
    for key,value in result[0].items():
        print(key,value.shape)
    for key,value in result[1].items():
        print(key,value.shape)
