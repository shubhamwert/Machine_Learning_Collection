import numpy as np
a=np.array([0,1,2,3,4,5])
a

a.ndim
#getting shape of array
b=(a.shape[0],a.ndim)
b

c=a.reshape((3,2))
c

a=np.array([1,2,3,4,5])

a>3

a[a<3]

a.clip(0,4)               #clip boundaries


import scipy as sp

scipy.version.full_version

scipy.dot==np.dot

#first Exercise
