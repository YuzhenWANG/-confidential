import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNN

# train_df=pd.read_excel('../input/train.xlsx')
# col_obj=[]
# for i in range(train_df.shape[1]):
#     if(train_df.iloc[:,i].dtype=='object'):
#         col_obj.append(str(train_df.iloc[:,i].name))
# print col_obj
# train_df_copy=train_df[col_obj]
# train_df_copy['Y']=train_df['Y']
# train_df_copy.to_csv('tool.csv')

# test=pd.read_excel('../input/test_A.xlsx')
# col_obj=[]
# for i in range(test.shape[1]):
#     if(test.iloc[:,i].dtype=='object'):
#         col_obj.append(str(test.iloc[:,i].name))
# print col_obj
# test=test[col_obj]
# test.to_csv('test_A.csv')

train_df=pd.read_csv('tool.csv')
train_df=train_df.drop(['ID'],axis=1)
train_copy=train_df.drop(['Y'],axis=1)
test_df=pd.read_csv('test_A.csv')
test_df=test_df.drop(['ID'],axis=1)
submit=pd.read_csv('../input/testA_sub.csv')
for i in range(test_df.shape[0]):
    print "row: ",i
    y=0
    cnt=0
    for j in range(train_copy.shape[0]):
        if(test_df.irow(i).all() == train_copy.irow(j).all()):
            y+=train_df.irow(j)['Y']
            cnt+=1
    submit.iloc[i,1]=y*1.0/cnt
    print y*1.0/cnt

submit.to_csv('submit.csv',index=False)

