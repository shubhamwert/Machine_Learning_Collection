import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv('sample_data.csv')

df.shape
plt.plot(df)
df.head()


#AutoMobileData

AutoMobileDataFrame=pd.read_csv('Automobile_data.csv')


AutoMobileDataFrame.head(10)
AutoMobileDataFrame.describe()

#working with missing data
AutoMobileDataFrame.count()
AutoMobileDataFrame.isnull().columns.value_counts()
AutoMobileDataFrame.isnull().sum()
AutoMobileDataFrame.fillna('0',mehtod='pad')
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


# AutoMobileDataFrame.replace(to_replace='?',value='nan',inplace=True)
# AutoMobileDataFrame.isna().sum(axis=0)
# for column in AutoMobileDataFrame.isnull().columns.values.tolist():
#     print(column)
#     print( AutoMobileDataFrame.isnull()[column].value_counts())