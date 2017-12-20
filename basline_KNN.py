from sklearn.neighbors import KNeighborsRegressor as KNN
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import random

# train_df=pd.read_csv('train_data.csv',index_col=0)
# print(train_df.shape)
# train_y = train_df['Y']
# train_df.drop(['Y'],axis=1,inplace=True)


# train_np = np.array(train_df)
# test_y = train_y[0:100]
# test_np = train_np[0:100]
# train_np = train_np[100::]
# train_y = train_y[100::]

# # V = np.cov(train_np.transpose())
# # kr = KNN(5,'distance',metric='mahalanobis',metric_params={'V':V})
# kr = KNN(5,'distance')
# kr.fit(train_np,train_y)

# t = kr.predict(test_np)
# d = t-test_y
# print((d*d).sum()/100)
test_np = pd.read_csv('testA.xlsx.pd',index_col=0)

train_df=pd.read_csv('train_data.csv',index_col=0)
print(train_df.shape)
train_y = train_df['Y']
train_df.drop(['Y'],axis=1,inplace=True)
train_np = np.array(train_df)
for i in range(train_np.shape[1]):
    temp = train_np[:,i]
    train_np[:,i] = (temp-temp.min())*1.0/(temp.max()-temp.min())

l = [i for i in range(500)]
random.shuffle(l)
test_num = 200
test_y = train_y[l[0:test_num]]
test_np = train_np[l[0:test_num]]

train_np = train_np[l[test_num:500]]
train_y = train_y[l[test_num:500]]

# test_y = train_y[0:100]
# test_np = train_np[0:100]

# train_np = train_np[100:500]
# train_y = train_y[100:500]

pc=PCA(n_components=0.94,whiten=True)  
pc = pc.fit(train_np)  
train_np = pc.transform(train_np)
test_np = pc.transform(test_np)
print(train_np.shape)

from sklearn import svm
# kr = KNN(17,'distance')
kr = svm.SVR()
kr.fit(train_np,train_y)
# model_SVR = svm.SVR()



t = kr.predict(test_np)
d = t-test_y
print((d*d).sum()/100)


t = kr.predict(train_np)
d = t-train_y
print((d*d).sum()/400)

# submit = pd.read_csv('templateA.csv')

# submit = submit.join(pd.DataFrame(t[1::]))
# submit.columns = [submit.columns[0],t[0]]
# # submit = pd.read_csv('templateB.csv')
# submit.to_csv('submit.csv',index=False)
