import numpy as np
import matplotlib.pyplot as plt
import model

X=np.random.rand(1000,1)
Y=np.power(X,2)
# Y=[1 if k >1.3 or k<1.1 else 0 for k in Y]
plt.scatter(X,Y)
X=np.concatenate((X,np.power(X,2)),axis=1)
# Y=np.asarray(Y)
print(Y.shape)
print(X.shape)
model=NN_1_layer(X.shape[1],Y.shape[1])
model.forward(X)
model.fit(X,Y,num_iters=1000)
y_hat=model.predict(X)
plt.scatter(X[:,0],y_hat)
plt.show()