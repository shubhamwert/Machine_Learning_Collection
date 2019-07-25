import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv('sample_data.csv')

df.shape
plt.plot(df)
df.describe()

!pip install tensorflow==2.0.0-alpha0 