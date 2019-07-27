import tensorflow as tf
from matplotlib import pyplot as plt
from random import randint

def train_mnist():
    
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.99):
                  print("\nReached 99% accuracy so cancelling training!")
                  self.model.stop_training = True
    

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    callbacks=myCallback()
    model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28,28)),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        x_train, y_train, epochs=10, callbacks=[callbacks]
    )
    x=0
    while(x<10):
            x=x+1 
            p=randint(1,100)
            plt.plot(x_test)
            print("label::",y_test[p],"\n predicted",model.predict(x_test[p]))
            input("\nPress Enter to continue...")


    return history.epoch, history.history['acc'][-1]

train_mnist()