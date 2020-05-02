import pandas as pd
from matplotlib import pyplot as plt
import random as rand
import numpy as np
df=pd.read_csv('sample_data.csv')



#AutoMobileData

AutoMobileDataFrame=pd.read_csv('Automobile_data.csv')


AutoMobileDataFrame.head(10)
AutoMobileDataFrame.describe()

#working with missing data
AutoMobileDataFrame.count()
AutoMobileDataFrame.isnull().columns.value_counts()
AutoMobileDataFrame.isnull().sum()
AutoMobileDataFrame.fillna('0')
df = pd.read_csv('Automobile_data.csv', na_values={
'price':["?","n.a"],
'stroke':["?","n.a"],
'horsepower':["?","n.a"],
'peak-rpm':["?","n.a"],
'average-mileage':["?","n.a"]})
print (df.isnull().sum())


df[['company','price']][df.price==df['price'].max()]


df[df.company=='toyota']


# alternative
d1=df.groupby('company')
d1.get_group('toyota')





df['company']. value_counts()

m=df.groupby('company')
m['company','price'].max()
df['price']=df['price'].replace('0',0)
df.cumsum(skipna=False)

for i in df.columns:

    plt.scatter(df[i],df['price'])
    plt.figure(figsize=(16,13))
    plt.xlabel(i)



df.head(5)



# AutoMobileDataFrame.replace(to_replace='?',value='nan',inplace=True)
# AutoMobileDataFrame.isna().sum(axis=0)
# for column in AutoMobileDataFrame.isnull().columns.values.tolist():
#     print(column)
#     print( AutoMobileDataFrame.isnull()[column].value_counts())
df=df.dropna()
p=df['price']
p=p.dropna()

#curve fitting linear
x=np.asarray(df['horsepower'])
y=p.to_numpy()
def LinearCurveFitting(x,y,alpha=0.01):
    x=Normalize(x)
    y=Normalize(y)
    theta=np.asarray([[0.1,0.2]])
    x=np.asarray([np.ones(len(x)),x])
    x
    theta   
    m=x.shape[0]
    for i in range(0,1000):
        h_theta=x.transpose().dot(theta.transpose())
        h_theta.shape
        y=np.asarray(y).transpose()
        y=np.reshape(y,(58,1))
        y.shape
        error=np.sum(np.square(h_theta-y))/(2*m)
        theta=theta-alpha*np.sum((x.transpose()*(h_theta-y)))/m
        
        print("for iteration ",i," J= ",error)
        plt.scatter(i,error)   
    return theta
def Normalize(X):
    return (X-np.mean(X))/np.std(X)

p=LinearCurveFitting(x,y,alpha=0.01)
plt.scatter(Normalize(x),Normalize(y),marker='x')
x1=np.asarray([np.ones(len(x)),Normalize(x)])
plt.scatter(Normalize(x),x1.transpose().dot(p.transpose()),marker='o')
