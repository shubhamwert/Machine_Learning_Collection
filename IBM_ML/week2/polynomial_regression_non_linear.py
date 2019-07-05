from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

def ModelSigmoid(x,a,b):
         y = 1 / (1 + np.exp(-a*(x-b)))
         return y



df=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv')

df.head()

plt.scatter(df['Year'],df['Value'],marker='x')


x_data, y_data = (df["Year"].values, df["Value"].values)
x_data =x_data/max(x_data)
y_data =y_data/max(y_data)
popt, pcov = curve_fit(ModelSigmoid, x_data, y_data)


plt.figure(figsize=(8,5))
y = ModelSigmoid(x_data, *popt)
plt.plot(x_data, y_data, 'ro', label='data')
plt.plot(x_data,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

