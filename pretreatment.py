#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pandas import Series, DataFrame
import seaborn as sns
sns.set(color_codes=True)
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.model_selection import  train_test_split


def MSE_score(predict_df,valid_df):
    if(len(predict_df)!=len(valid_df)):
        print "The length is not same !"
    else:
        sum=0
        for i in range(len(predict_df)):
            sum+=(predict_df[i]-valid_df.ravel()[i])*(predict_df[i]-valid_df.ravel()[i])
    return sum*1.0/len(predict_df)



# # 1: To delete the columns where variance is 0
# train_df=pd.read_excel('../input/train.xlsx')
# print train_df.head()
# print "done loading data"
# list=[]
# list_obj=[]
# var=[]
# print train_df.info()
# for i in range(train_df.shape[1]):
#     print "col number: ",i
#     if(train_df.icol(i).dtype!='object' and np.var(train_df.icol(i))==0):
#         list.append(str(train_df.icol(i).name))
#     elif (train_df.icol(i).dtype == 'object'):
#         list_obj.append(i)
# print len(list)
# print list
#
# for label in list:
#     train_df.drop([label],axis=1,inplace=True)
# print train_df.shape
# train_df.to_csv('../input/train_without_var0')



# # 2: To delete the columns where variance is too small (variance<10)
# train_df=pd.read_csv('../input/train_without_var0')
# target=train_df['Y']
# id=train_df['ID']
# print type(target)
# var=[]
# list=[]
# for i in range(train_df.shape[1]):
#     if(train_df.iloc[:,i].dtype!='object' and np.var(train_df.iloc[:,i])<1):
#         var.append(np.var(train_df.iloc[:,i]))
#         list.append(train_df.iloc[:,i].name)
#     elif (train_df.icol(i).dtype == 'object'):
#         list.append(train_df.icol(i).name)
# print var
# print len(var)
# for label in list:
#     train_df.drop([label],axis=1,inplace=True)
# print train_df.shape
#
#
# train_df['Y']=target
# train_df['ID']=id
# train_df.to_csv('../input/train_without_smallVar')



# # 3: To delete columns where NA ratio > 0.95
# def FindFeatureNAorValue(data, feature_cols, axis=0, value = 'NA', prob_dropFct = 0.95):
#     '''
#     函数说明：寻找每一个特征有多少value值，默认为：缺失值，及所占比率
#     输入：data——整个数据集，包括Index，target
#         feature_cols——特征名
#         prob_dropFct——大于这个比例，就丢掉该特征
#     输出：numValue——DataFrame  index='feature1', columns=['numnumValue', 'probnumValue']
#         dropFeature_cols——要丢掉的特征列名
#     '''
#     #计算x中value值个数
#     def num_Value(x, value = 'NA'):
#         if value == 'NA':
#             return sum(x.isnull())   #寻找缺失值个数
#         else:
#             return sum(x == value)  #寻找某个值value个数
#
#     numValue = data[feature_cols].apply(num_Value, axis=axis,args=[value])
#     numValue = DataFrame(numValue, columns = ['numValue'])
#     nExample = data.shape[0]
#     probValue = map(lambda x: round(float(x)/nExample, 4), numValue['numValue'])
#     numValue['probValue'] = probValue
#
#
#     #寻找缺失值大于prob_dropFct的特征
#     dropFeature = numValue[numValue['probValue'] >= prob_dropFct]
#     dropFeature_cols = list(dropFeature.index)
#
#     return numValue,dropFeature_cols
#
#
#
# train_df=pd.read_csv('../input/train_without_smallVar')
# col_list=[column for column in train_df]
# numValue,dropFeature_cols=FindFeatureNAorValue(train_df,col_list)
# print len(dropFeature_cols)
# train_df=train_df.drop(dropFeature_cols,axis=1)
# train_df.to_csv('../input/train_without_NAN')



# 4: To delete rows where NA ratio > X
train_df=pd.read_csv('../input/train_without_NAN')  # 405 rows have NA value
def num_Value(x, value = 'NA'):
    if value == 'NA':
        return sum(x.isnull())   #寻找缺失值个数
    else:
        return sum(x == value)  #寻找某个值value个数
row_del=[]
for i in range(train_df.shape[0]):
    if(num_Value(train_df.iloc[i,:])*1.0/train_df.shape[1]>0/9):
        row_del.append(i)
print row_del
print len(row_del)