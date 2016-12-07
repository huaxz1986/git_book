import numpy as np
import pandas as pd
import  pickle
import  time
import os
def current_time():
    '''
    以固定格式打印当前时间

    :return:返回当前时间的字符串
    '''
    return time.strftime('%Y-%m-%d %X', time.localtime())
class Data_Cleaner:
    '''
    数据清洗器

    它的初始化需要提供三个文件的文件名。它提供了唯一的对外接口：load_data()。它返回清洗好的数据。
    如果数据已存在，则直接返回。否则将执行一系列清洗操作并返回清洗好的数据。
    '''
    def __init__(self,people_file_name,act_train_file_name,act_test_file_name):
        '''

        :param people_file_name: people.csv文件的 file_path
        :param act_train_file_name: act_train.csv文件的 file_path
        :param act_test_file_name:act_test.csv文件的 file_path
        :return:
        '''
        self.p_fname=people_file_name
        self.train_fname=act_train_file_name
        self.test_fname=act_test_file_name
        self.types=['type %d'%i for i in range(1,8)]
        self.fname='output/cleaned_data'
    def load_data(self):
        '''
        加载清洗好的数据

         如果数据已经存在，则直接返回。如果不存在，则加载 csv文件，然后合并数据、拆分成 type1 ~type7，然后执行数据类型转换，
        最后重新排列每个列的顺序。然后保存数据并返回数据。

        :return:一个元组：依次为：self.train_datas,self.test_datas
        '''
        if(self._is_ready()):
            print("cleaned data is availiable!\n")
            self._load_data()
        else:
            self._load_csv()
            self._merge_data()
            self._split_data()
            self._typecast_data()
            self._save_data()
        return self.train_datas,self.test_datas

    def _load_csv(self):
        '''
        加载 csv 文件

        :return:
        '''
        print("----- Begin run load_csv at %s -------"%current_time())
        self.people=pd.read_csv(self.p_fname,sep=',',header=0,keep_default_na=True,parse_dates=['date'])
        self.act_train=pd.read_csv(self.train_fname,sep=',',header=0,keep_default_na=True,parse_dates=['date'])
        self.act_test=pd.read_csv(self.test_fname,sep=',',header=0,keep_default_na=True,parse_dates=['date'])

        self.people.set_index(keys=['people_id'],drop=True,append=False,inplace=True)
        self.act_train.set_index(keys=['people_id'],drop=True,append=False,inplace=True)
        self.act_test.set_index(keys=['people_id'],drop=True,append=False,inplace=True)

        print("----- End run load_csv at %s -------"%current_time())
    def _merge_data(self):
        '''
        合并 people 数据和 activity 数据

        :return:
        '''
        print("----- Begin run merge_data at %s -------"%current_time())
        self.train_data=self.act_train.merge(self.people,how='left',left_index=True,right_index=True,suffixes=('_act', '_people'))
        self.test_data=self.act_test.merge(self.people,how='left',left_index=True,right_index=True,suffixes=('_act', '_people'))
        print("----- End run merge_data at %s -------"%current_time())
    def _split_data(self):
        '''
        拆分数据为 type 1~ 7

        :return:
        '''
        print("----- Begin run split_data at %s -------"%current_time())
        self.train_datas={}
        self.test_datas={}
        for _type in self.types:
            ## 拆分
            self.train_datas[_type]=self.train_data[self.train_data.activity_category==_type].dropna(axis=(0,1), how='all')
            self.test_datas[_type]=self.test_data[self.test_data.activity_category==_type].dropna(axis=(0,1), how='all')
            # 删除列 activity_category
            self.train_datas[_type].drop(['activity_category'],axis=1,inplace=True)
            self.test_datas[_type].drop(['activity_category'],axis=1,inplace=True)
            # 将列 activity_id 作为索引
            self.train_datas[_type].set_index(keys=['activity_id'], drop=True, append=True, inplace=True)
            self.test_datas[_type].set_index(keys=['activity_id'], drop=True, append=True, inplace=True)
        print("----- End run split_data at %s -------"%current_time())

    def _typecast_data(self):
        '''
        执行数据类型转换，将所有数据转换成浮点数

        :return:
        '''
        print("----- Begin run typecast_data at %s -------"%current_time())
        str_col_list=['group_1']+['char_%d_act'%i for i in range(1,11)]+['char_%d_people'%i for i in range(1,10)]
        bool_col_list=['char_10_people']+['char_%d'%i for i in range(11,38)]

        for _type in self.types:
            for data_set in [self.train_datas,self.test_datas]:
                # 处理日期列
                data_set[_type].date_act= (data_set[_type].date_act- np.datetime64('1970-01-01'))/ np.timedelta64(1, 'D')
                data_set[_type].date_people= (data_set[_type].date_people- np.datetime64('1970-01-01'))/ np.timedelta64(1,'D')
                # 处理 group 列
                data_set[_type].group_1=data_set[_type].group_1.str.replace("group",'').str.strip().astype(np.float64)
                # 处理布尔值列
                for col in bool_col_list:
                    if col in data_set[_type]:data_set[_type][col]=data_set[_type][col].astype(np.float64)
                # 处理其他字符串列
                for col in str_col_list[1:]:
                    if col in data_set[_type]:data_set[_type][col]=data_set[_type][col].str.replace("type",'').str.strip().astype(np.float64)

            data_set[_type]= data_set[_type].astype(np.float64)
        print("----- End run typecast_data at %s -------"%current_time())
    def _is_ready(self):
        if(os.path.exists(self.fname)):
            return True
        else :
            return False
    def _save_data(self):
        print("----- Begin run save_data at %s -------"%current_time())
        with open(self.fname,"wb") as file:
            pickle.dump([self.train_datas,self.test_datas],file=file)
        print("----- End run save_data at %s -------"%current_time())
    def _load_data(self):
        print("----- Begin run _load_data at %s -------"%current_time())
        with open(self.fname,"rb") as file:
            self.train_datas,self.test_datas=pickle.load(file)
        print("----- End run _load_data at %s -------"%current_time())

if __name__=='__main__':
    clearner=Data_Cleaner("./data/people.csv",'./data/act_train.csv','./data/act_test.csv')
    result=clearner.load_data()
    for key,item in result[0].items():
        for col in item.columns:
            unique_value=item[col].unique()

            if(len(unique_value)<=100):
                print(col,':len=',len(unique_value),'\t;data=',unique_value)
            else:print(col,':len=',len(unique_value))

        print("\n=======\n")



