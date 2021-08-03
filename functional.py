import tensorflow as tf
import numpy
epoch = 5
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
"""
x_train = x_train[0:2500]
x_test = x_test[0:2500]

y_train = y_train[0:2500]
y_test = y_test[0:2500]
"""
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

img_inputs = tf.keras.Input(shape=(28,28,1))  
x = tf.keras.layers.Flatten()(img_inputs)
x = tf.keras.layers.Dense(784, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=img_inputs, outputs=output, name="Functional")

model.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # The loss uses SparseCategoricalCrossentropy, it does't need to convert the label to onehot matrix
              optimizer= 'adam',
              metrics=['accuracy'])
h = model.fit(x_train, y_train, epochs=epoch)

predictions = model.predict(x_test)
print(numpy.argmax(predictions[3]))
print(y_test[3])


