import numpy as np
class BAM:
    def __init__(self):
        pass
    def fit(self,X: np.array,Y:np.array):
        self.W=np.asmatrix(np.zeros((len(X[0]),len(Y[0]))))
        if(len(X)!=len(Y)):
            print("ERROR length of X not equal to Y")
        else :
            for i in range(len(X)):
                
                p=np.asmatrix(X[i])
                p=p.transpose()
                
                self.W=self.W+p*Y[i]
        
    def __repr__(self):

            
        try :
            print("BAM with weights {}".format(self.W))
            return super().__repr__()
        except:
            return super().__repr__()

    def recover_for_X(self,X):
        Y1=X*self.W
        return self.signum(Y1[0,:])

    def signum(self,P: np.array):
        Y=[]
        print(len(P))
        Y=(np.asarray(P)>0)

        return np.asarray(Y)

   



            



