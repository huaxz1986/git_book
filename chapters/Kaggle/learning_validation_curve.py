import  numpy as np
import  matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve,validation_curve

from sklearn.model_selection import train_test_split
from data_clean import current_time
from data_preprocess import Data_Preprocesser,Data_Cleaner
def cut_data(data,scale_factor,stratify=True,seed=0):
    '''
    切分数据集，使用其中的一部分来学习

    :param data:原始数据集
    :param scale_factor:传递给 train_test_split 的 train_size 参数，可以为浮点数([0,1.0])，可以为整数
    :param stratify:传递给 train_test_split 的 stratify 参数
    :param seed: 传递给 train_test_split 的 seed 参数
    :return: 返回一部分数据集
    '''
    if stratify:
            return train_test_split(data,train_size=scale_factor,stratify=data[:,-1],random_state=seed)[0]
    else:
            return train_test_split(data,train_size=scale_factor,random_state=seed)[0]
class Curver_Helper:
    '''
    学习曲线和验证曲线的辅助类，用于保存曲线和绘制曲线

    '''
    def __init__(self,curve_name,xlabel,x_islog):
        '''
        初始化函数

        :param curve_name:曲线名称
        :param xlabel:曲线 X轴的名称
        :param x_islog:曲线 X轴是否为对数
        :return:
        '''
        self.curve_name=curve_name
        self.xlabel=xlabel
        self.x_islog=x_islog
    def save_curve(self,x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std):
        '''
        保存曲线的数据

        :param x_data: 曲线的 x 轴数据，也就是被考察的指标的序列
        :param train_scores_mean: 训练集预测的平均得分
        :param train_scores_std:训练集预测得分的标准差
        :param test_scores_mean:测试集预测的平均得分
        :param test_scores_std:测试集预测得分的标准差
        :return:
        '''
        with open("output/%s"%self.curve_name,"wb") as output:
            result_array=np.array([x_data,train_scores_mean,train_scores_std,
                                   test_scores_mean,test_scores_std])
            np.save(output,result_array)
    def plot_curve(self,x_data,train_scores_mean,train_scores_std
                           ,test_scores_mean,test_scores_std):
            '''
            绘图并保存图片

            :param x_data:曲线的 x 轴数据，也就是被考察的指标的序列
            :param train_scores_mean:训练集预测的平均得分
            :param train_scores_std:训练集预测得分的标准差
            :param test_scores_mean:测试集预测的平均得分
            :param test_scores_std:测试集预测得分的标准差
            :return:
            '''
            min_y1=np.min(train_scores_mean)
            min_y2=np.min(test_scores_mean)
            fig=plt.figure(figsize=(20,15))
            ax=fig.add_subplot(1,1,1)
            ax.plot(x_data, train_scores_mean, label="Training roc_auc", color="r",marker='o')
            ax.fill_between(x_data, train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std, alpha=0.2, color="r")
            ax.plot(x_data, test_scores_mean, label="Testing roc_auc", color="g",marker='+')
            ax.fill_between(x_data, test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std, alpha=0.2, color="g")
            ax.set_title("%s"%self.curve_name)
            ax.set_xlabel("%s"%self.xlabel)
            ax.locator_params(axis='x', tight=True, nbins=10)
            ax.grid(which="both")
            if self.x_islog:ax.set_xscale('log')
            ax.set_ylabel("Score")
            ax.set_ylim(min(min_y1,min_y2)-0.1,1.1)
            ax.set_xlim(0,max(x_data))
            ax.legend(loc='best')
            ax.grid(True,which='both',axis='both')
            fig.savefig("output/%s.png"%self.curve_name,dpi=100)
    @classmethod
    def plot_from_saved_data(self,file_name,curve_name,xlabel,x_islog):
        '''
        通过保存的数据点来绘制并保存图形

        :param file_name: 保存数据点的文件名
        :param curve_name:曲线名称
        :param xlabel:曲线 X轴的名称
        :param x_islog:曲线 X轴是否为对数
        :return:
        '''
        x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std=np.load(file_name)
        helper=Curver_Helper(curve_name,xlabel,x_islog)
        helper.plot_curve(x_data,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std)
class Curver:
    '''
    用于生成学习曲线验证曲线的父类
    '''
    def create_curve(self,train_data,curve_name,xlabel,x_islog,scale=0.1,is_gui=False):
        '''
        生成曲线

        :param train_data:训练数据集
        :param curve_name : 曲线名字，用于绘图和保存文件
        :param xlabel: 曲线 X轴名字
        :param x_islog: X轴是否为 对数坐标
        :param scale:切分比例，默认使用 10%的训练集
        :param is_gui:是否在 GUI环境下。如果在 GUI环境下，则绘制图片并保存
        :return:
        '''
        class_name=self.__class__.__name__
        self.curve_name=curve_name
        ###  加载数据
        data=cut_data(train_data,scale_factor=scale,stratify=True,seed=0)
        self.X=data[:,:-1]
        self.y=data[:,-1]
        ##### 生成曲线参数 ###
        result=self._curve()
        ####### 保存和绘制曲线 #######
        self.helper=Curver_Helper(self.curve_name,xlabel,x_islog)
        if(is_gui):self.helper.plot_curve(*result)
        self.helper.save_curve(*result)
class LearningCurver(Curver):
    def __init__(self,train_sizes):
        self.train_sizes=train_sizes
        self.estimator=GradientBoostingClassifier(max_depth=10)

    def _curve(self):
        print("----- Begin run learning_curve(%s) at %s -------"%(self.curve_name,current_time()))
        #### 获取学习曲线 ######
        abs_trains_sizes,train_scores, test_scores = learning_curve(self.estimator,
                self.X, self.y,cv=3,scoring="roc_auc",train_sizes=self.train_sizes,n_jobs=-1,verbose=1)
        print("----- End run learning_curve(%s) at %s -------"%(self.curve_name,current_time()))
        ###### 对每个 test_size ，获取 3 折交叉上的预测得分上的均值和方差 #####
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        return abs_trains_sizes,train_scores_mean,train_scores_std,test_scores_mean,test_scores_std
class ValidationCurver(Curver):
    def __init__(self,param_name,param_range):
        self.p_name=param_name
        self.p_range=param_range
        self.estimator=GradientBoostingClassifier()
    def _curve(self):
        print("----- Begin run validation_curve(%s) at %s -------"%(self.curve_name,current_time()))
        train_scores, test_scores = validation_curve(self.estimator, self.X, self.y, param_name=self.p_name,
                param_range=self.p_range,cv=3, scoring="roc_auc",n_jobs=-1,verbose=1)
        print("----- End run validation_curve(%s) at %s -------"%(self.curve_name,current_time()))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        return [item for item in self.p_range],train_scores_mean,train_scores_std,test_scores_mean,test_scores_std

def run_learning_curve(data,type_name):
    '''
    生成学习曲线
	
    :param data: 训练集
    :param type_name：数据种类名
    :return:
    '''
    learning_curver=LearningCurver(train_sizes=np.logspace(-1,0,num=10,endpoint=True,dtype='float'))
    learning_curver.create_curve(data,"learning_curve_%s"%type_name,
                                 xlabel="Nums",x_islog=True,scale=0.99,is_gui=True)
def run_test_subsample(data,type_name,scale,param_range):
    '''
    生成验证曲线，验证 subsample 参数
	
    :param data: 训练集
    :param type_name：数据种类名
	:param scale: 样本比例，一个小于1.0的浮点数
	:param param_range: subsample 参数的范围
    :return:
    '''
    validation_curver=ValidationCurver("subsample",param_range=param_range)
    validation_curver.create_curve(data,"validation_curve_subsample_%s"%type_name,
                                   xlabel='subsample',x_islog=False,scale=scale,is_gui=True)
def run_test_n_estimators(data,type_name,scale,subsample,param_range):
    '''
    生成验证曲线，验证 n_estimators 参数
	
    :param data: 训练集
    :param type_name：数据种类名
	:param scale: 样本比例，一个小于1.0的浮点数
	:param subsample: subsample参数
	:param param_range: n_estimators 参数的范围
	:return:
    '''
	
    validation_curver=ValidationCurver("n_estimators",param_range=param_range)
    validation_curver.estimator.set_params(subsample=subsample)#调整 subsample
    validation_curver.create_curve(data,"validation_curve_n_estimators_%s"%type_name
                                   ,xlabel='n_estimators',x_islog=True,scale=scale,is_gui=True)
def run_test_maxdepth(data,type_name,scale,subsample,n_estimators,param_range):
    '''
    生成验证曲线，验证 maxdepth 参数
	
    :param data: 训练集
    :param type_name：数据种类名
	:param scale: 样本比例，一个小于1.0的浮点数
	:param subsample: subsample参数
	:param n_estimators: n_estimators 参数
	:param param_range: maxdepth 参数的范围
	:return:
    '''
    validation_curver=ValidationCurver("max_depth",param_range=param_range)
    validation_curver.estimator.set_params(subsample=subsample) # 调整 subsample
    validation_curver.estimator.set_params(n_estimators=n_estimators) # 调整 n_estimators
    validation_curver.create_curve(data,"validation_curve_maxdepth_%s"%type_name
                                   ,xlabel='maxdepth',x_islog=True,scale=scale,is_gui=True)

if __name__=="__main__":
    clearner=Data_Cleaner("./data/people.csv",'./data/act_train.csv','./data/act_test.csv')
    result=clearner.load_data()
    preprocessor=Data_Preprocesser(*result)
    train_datas,test_datas=preprocessor.load_data()

    #############  创建学习曲线 ############
    # run_learning_curve(train_datas['type 7'],'type7')
    ###########   创建验证曲线   #######
    ### 验证 subsample ，需要学习曲线的结果
    # run_test_subsample(train_datas['type 7'],'type7',0.5,param_range=np.linspace(0.01,1,num=10,dtype=float))
    # run_test_n_estimators(train_datas['type 7'],'type7',0.5,0.4,param_range=np.logspace(0,3.5,num=10,dtype=int))
    run_test_maxdepth(train_datas['type 7'],'type7',0.5,0.4,35,param_range=np.logspace(0,3,num=10,dtype=int))
