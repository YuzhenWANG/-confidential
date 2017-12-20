import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.preprocessing import Normalizer
import sys
import json

def preprocess(file_name,out_file_name):
    # 1: To delete the columns where variance is 0
    # 2: To delete the columns whose type is object
    final_drop_list = []
    print("To delete the columns where variance is 0")
    print("To delete the columns whose type is object")
    train_df=pd.read_excel(file_name)
    print(train_df.head())
    print("done loading data")
    drop_lists1=[]
    print(train_df.info())

    for i in range(train_df.shape[1]):
        #print("col number: ",i)
        if train_df.iloc[:,i].dtype!='object' and train_df.iloc[:,i].max()==train_df.iloc[:,i].min():
            drop_lists1.append(str(train_df.iloc[:,i].name))
        elif (train_df.iloc[:,i].dtype == 'object'):
            drop_lists1.append(str(train_df.iloc[:,i].name))
    print(len(drop_lists1))
    # print(train_df)

    train_df1 = train_df.drop(drop_lists1,axis=1)
    print(train_df1.shape)
    final_drop_list.extend(drop_lists1)
    # train_df1.to_csv('./input/1.csv')

    # 3: To delete columns where NA ratio > 0.95
    print("3: To delete columns where NA ratio > 0.95")
    def FindFeatureNAorValue(data, feature_cols, axis=0, value = 'NA', prob_dropFct = 0.95):
        '''
        函数说明：寻找每一个特征有多少value值，默认为：缺失值，及所占比率
        输入：data——整个数据集，包括Index，target
            feature_cols——特征名
            prob_dropFct——大于这个比例，就丢掉该特征
        输出：numValue——DataFrame  index='feature1', columns=['numnumValue', 'probnumValue']
            dropFeature_cols——要丢掉的特征列名
        '''
        #计算x中value值个数
        def num_Value(x, value = 'NA'):
            if value == 'NA':
                return sum(x.isnull())   #寻找缺失值个数
            else:
                return sum(x == value)  #寻找某个值value个数

        numValue = data[feature_cols].apply(num_Value, axis=axis,args=[value])
        numValue = DataFrame(numValue, columns = ['numValue'])
        nExample = data.shape[0]
        probValue = list( map(lambda x: round(float(x)/nExample, 4), numValue['numValue']))
        numValue['probValue'] = probValue


        #寻找缺失值大于prob_dropFct的特征
        dropFeature = numValue[numValue['probValue'] >= prob_dropFct]
        dropFeature_cols = list(dropFeature.index)

        return numValue,dropFeature_cols

    # train_df1=pd.read_csv('./input/1.csv')
    col_list=[column for column in train_df1]
    numValue,dropFeature_cols=FindFeatureNAorValue(train_df1,col_list)
    print(len(dropFeature_cols))
    train_df2 = train_df1.drop(dropFeature_cols,axis=1)
    print(train_df2.shape)
    # train_df2.to_csv('./input/2.csv')
    final_drop_list.extend(dropFeature_cols)

    # 4: To delete columns where variance is too small
    print("4: To delete columns where variance is too small")
    # train_df2=pd.read_csv('./input/2.csv')
    drop_lists3=[]
    for i in range(train_df2.shape[1]):
        if(train_df2.iloc[:,i].dtype!='object'):
            temp = train_df2.iloc[:,i]
            temp = (temp-temp.min())*1.0/(temp.max()-temp.min())
            vt = temp.var()
            if vt < 0.03:
                drop_lists3.append(train_df2.iloc[:,i].name)
    print(len(drop_lists3))

    train_df3 = train_df2.drop(drop_lists3,axis=1)
    print(train_df3.shape)
    final_drop_list.extend(drop_lists3)

    # train_df3.to_csv('./input/3.csv')


    # 5: Fill NA 
    print("5: Fill NA")
    # train_df3=pd.read_csv('./input/3.csv')
    natable={}
    print(train_df3.shape)
    for i in range(train_df3.shape[1]):
        if(train_df3.iloc[:,i].dtype!='object'):
            temp = train_df3.iloc[:,i]
            meanofcol = temp.mean()
            natable[train_df3.iloc[:,i].name] = meanofcol

    train_df4 = train_df3.fillna(value=natable)
    print(train_df4.shape)
    train_df4.to_csv(out_file_name)
    return train_df4,final_drop_list


def main():
    _,droplist = preprocess(sys.argv[1],sys.argv[2])
    with open("%s.drop_list"%sys.argv[2],'w') as fp:
        json.dump(droplist,fp)
    
def main_t():
    premodel = {}
    with open("./input/train_data.xlsx.drop_list.pd",'r') as fp:
        premodel = json.load(fp)
    test_df=pd.read_excel(sys.argv[1])
    test_df.drop(premodel['droplist'],axis=1,inplace=True)

    natable={}
    for i in range(test_df.shape[1]):
        if(test_df.iloc[:,i].dtype!='object'):
            temp = test_df.iloc[:,i]
            meanofcol = temp.mean()
            natable[test_df.iloc[:,i].name] = meanofcol
    test_df = test_df.fillna(value=natable)

    for i in range(test_df.shape[1]):
        temp = test_df.iloc[:,i]
        test_df.iloc[:,i] = (temp-premodel['min'][i]*1.0)/premodel['scale'][i]
    test_df.to_csv(sys.argv[1]+'.pd')

if __name__ == '__main__':
    main_t()
    #main()

