import  scipy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from data_clean import current_time
from sklearn.model_selection import train_test_split
from data_preprocess import Data_Preprocesser,Data_Cleaner

def grid_search(tuned_parameters,data,train_size,seed):
    '''
    参数优化

    :param tuned_parameters: 待优化的参数字典
    :param data: 数据集
    :param train_size:训练集大小
    :param seed:用于生成随机数种子
    :return:
    '''

    print("----- Begin run grid_search at %s -------"%current_time())
    X=data[:,:-1]
    y=data[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train_size,stratify=data[:,-1],random_state=seed)
    clf=GridSearchCV(GradientBoostingClassifier(),tuned_parameters,cv=10,scoring="roc_auc")
    clf.fit(X_train,y_train)
    print("Best parameters set found:",clf.best_params_)
    print("Randomized Grid scores:")
    for params, mean_score, scores in clf.grid_scores_:
        print("\t%0.3f (+/-%0.03f) for %s" % (mean_score, scores.std() * 2, params))
        print("Optimized Score:",clf.score(X_test,y_test))
        print("Detailed classification report:")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
    print("----- End run grid_search at %s -------"%current_time())

if __name__=='__main__':
    clearner=Data_Cleaner("./data/people.csv",'./data/act_train.csv','./data/act_test.csv')
    result=clearner.load_data()
    preprocessor=Data_Preprocesser(*result)
    train_datas,test_datas=preprocessor.load_data()
    tuned_parameters={'subsample':[0.3,0.35,0.4,0.45,0.5,0.55,0.6],
                      'n_estimators':[30,35,50,100,150,200]
        ,
                      'max_depth':[2,4,8,16,32]}
    grid_search(tuned_parameters,train_datas['type 7'],train_size=0.75,seed=0)