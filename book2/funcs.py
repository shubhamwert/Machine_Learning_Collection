import math
import numpy as np
def pow_transformation(y: np.array,lamda):
    
    if(lamda!=0):
        yw=np.power(y,lamda)-1
        yw=yw/lamda
    else:
        yw=np.log(y)
    return yw

