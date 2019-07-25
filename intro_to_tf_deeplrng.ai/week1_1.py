import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
model =keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])]) # Your Code Here#
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = list(range(0,10))
ys=list(range(0,10))
for i in xs:
  ys[i]=xs[i]*50
plt.plot(xs,ys)
model.fit(xs,ys,epochs=2500)
print(model.predict(list(range(10,20))))
plt.plot(model.predict(list(range(1,10))))
plt.show()