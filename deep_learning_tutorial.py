import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale numbers between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 'neurons';relu is a good default activation function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer: has # of classifications (10)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)  # training the model

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('epic_num_reader.model')  # how to save a model
new_model = tf.keras.models.load_model('epic_num_reader.model')  # how to load a model again

predictions = new_model.predict([x_test])


plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
#print(x_train[0])
