import numpy as np
from BAM_network import BAM


if __name__ == "__main__":
    X=np.asarray([[1,1,1,-1,1],[-1,1,1,-1,1],[1,1,-1,-1,1,]])
    Y=np.asarray([[1,1,-1],[-1,-1,-1],[1,1,1]])

    X=[[1,-1,1,-1,1,-1],[1,1,1,-1,-1,-1]]
    Y=[[1,1,-1,-1],[1,-1,1,-1]]


    model=BAM()
    model.fit(X,Y)
    Y_p=model.recover_for_X(X[0])
    print(Y_p)
    print(Y[0])
    


