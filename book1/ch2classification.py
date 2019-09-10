import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

data=load_iris()
x=data.data
x_names=data.feature_names
target = data.target 
target_names = data.target_names

for t in range(3):
    if t==0:
        c='r'
        marker='*'
    if t==1:
        c='b'
        marker='x'
    if t==2:
        c='g'
        marker='o'
    plt.scatter(x[target==t,0],x[target==t,1],marker=marker,c=c)
    plt.xlabel(x_names[0])
    plt.ylabel(x_names[1])    

for t in range(3):
    if t==0:
        c='r'
        marker='*'
    if t==1:
        c='b'
        marker='x'
    if t==2:
        c='g'
        marker='o'
    plt.scatter(x[target==t,1],x[target==t,2],marker=marker,c=c)
    plt.xlabel(x_names[1])
    plt.ylabel(x_names[2])    
for t in range(3):
    if t==0:
        c='r'
        marker='*'
    if t==1:
        c='b'
        marker='x'
    if t==2:
        c='g'
        marker='o'
    plt.scatter(x[target==t,2],x[target==t,3],marker=marker,c=c)
    plt.xlabel(x_names[2])
    plt.ylabel(x_names[3])    

#okay now lets classify

labels=target_names[target]
is_setosa=(labels=='setosa')

is_not_setosa=~is_setosa
plength=x[:,2]
max_setosa=plength[is_setosa].max()
min_non_setosa=plength[is_not_setosa].min()
#okay now a simple model::

#if petal length closer to 1.9 it is setosa and if it is closer to 3 it is not setosa

avg=(max_setosa+min_non_setosa)/2.0

k=(plength<2)
p=(k==is_setosa)
print("accuracy is equal to ",sum(p==True)/sum(p))


#our model did very good
# lets visualize it

 
for t in range(3):
    if t==0:
        c='r'
        marker='*'
    if t==1:
        c='b'
        marker='x'
    if t==2:
        c='g'
        marker='o'
    plt.scatter(range(0,len(x[target==t,2])),x[target==t,2],marker=marker,c=c)
    plt.xlabel('petal length')
    plt.ylabel('')
    #now our classifier
    #visualiztion    
    plt.plot(range(0,50),np.full((50,),avg))

   
    
#Cross validation and Testing
