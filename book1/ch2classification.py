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
