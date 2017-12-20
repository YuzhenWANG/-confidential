import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.preprocessing import Normalizer


def preprocess(file_name):
    # 1: To delete the columns where variance is 0
    # 2: To delete the columns whose type is object
    print("To delete the columns where variance is 0")
    print("To delete the columns whose type is object")
    train_df=pd.read_excel(file_name)
    print(train_df.head())
    print("done loading data")
    drop_lists1=[]
    print(train_df.info())
    target=train_df['Y']
    print(type(target))

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

    # train_df3.to_csv('./input/3.csv')


    # 5: Fill NA and delete obscure columns
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
    # train_df4.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1'],axis=1,inplace=True)
    print(train_df4.shape)
    train_df4.to_csv('./input/process.csv')


def main():
    preprocess('./input/train_data.xlsx')

if __name__ == '__main__':
    main()

