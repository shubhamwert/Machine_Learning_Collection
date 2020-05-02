import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import random as rand
data=sc.genfromtxt()

def error(f,x,y):
    return  sc.sum((f(x)-y)**2)

#polyfit_function

def drawPoly(fp1,x=range(1,100)):
    y1=0
    i=len(fp1)-1
    
    while i >   -1:
        y1=y1+(x**i)*fp1[len(fp1)-i-1]
        i=i-1



    plt.scatter(x,y1)
    return y1





if __name__ == "__main__":
    x=np.array(range(1,100))
    y=x-x**2-x**3+x**4                                  #adding function
    y=y+y*np.sin(x)/10
    plt.scatter(x,y,marker='x')


    fp1, residuals, rank, sv, rcond = sc.polyfit(x, y, 4, full=True)
    #therefore our f(x) is fp1
    print(fp1)
    residuals
    drawPoly(fp1,x)
    def f(x):
        y1=0
        i=len(fp1)-1
    
        while i >   -1:
            y1=y1+(x**i)*fp1[len(fp1)-i-1]
            i=i-1
        return y1
    
    

    


