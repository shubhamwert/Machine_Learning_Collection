# A more complex classifier
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
dataFrame=pd.read_csv("seeds_dataset.txt", sep="\t",header=None,names=["AREA","PERIMETER","COMPACTNESS","length of kernel","width of kernel","asymmetry coefficient","length of kernel groove","target"] )
dataFrame.head()
dataFrame.tail(19)

#some data is wrong so we will drop them
dataFrame=dataFrame.dropna()
features=dataFrame.columns
x=dataFrame[features[0:-1]]
x.shape
x.head()
y=dataFrame[features[-1]]
x.isna().sum()
y.isna().sum()

for t in range(1,5):
    
    if t==1:
        c='r'
        marker='x'
    if t==2:
        c='g'
        marker='o'
    if t==3:
        c='b'
        marker='*'
    if t==4:
        c='y'
        marker='+'
    plt.scatter(x[features[0]][y==t],x[features[1]][y==t].values,c=c,marker=marker)

for t in range(1,5):
    
    if t==1:
        c='r'
        marker='x'
    if t==2:
        c='g'
        marker='o'
    if t==3:
        c='b'
        marker='*'
    if t==4:
        c='y'
        marker='+'
    plt.scatter(x[features[5]][y==t],x[features[6]][y==t].values,c=c,marker=marker)
# so there is no particluar classification layer clearly visible
x.corr()
sb.heatmap(x.corr())

Myclassifier=KNeighborsClassifier()

p=KFold(n_splits=5,shuffle=True)
means=[]


for training,testing in p.split(x):
    
    Myclassifier.fit(x[training],y[training])
    prediction=Myclassifier.predict(features[testing])
    currmean=sc.mean(prediction==y[testing])
    means.append(currmean)
print("Mean accuracy: {:.1%}".format(sc.mean(means)))