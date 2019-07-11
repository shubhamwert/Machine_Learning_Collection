import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import r2_score
df=pd.read_csv('sample_data\data.csv')
df.head()

df.columns
df.isnull()

df.columns

X=df[['Urban population', 'Late births ', 'Wine consumption','Liquor consumption']]

Y=df[['Cirrhosis death rate']]
Y

plt.scatter(X['Late births '],Y)
plt.subplot()
plt.scatter(X['Urban population'],Y,color='red')
plt.scatter(X['Wine consumption'],Y,color='violet')
plt.scatter(X['Liquor consumption'],Y,color='yellow')

model=lr()
model.fit(X,Y)
y_pred=model.predict(X)
plt.plot(Y)
plt.plot(y_pred)

print(r2_score(Y,y_pred))






