from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import plot_model

import numpy as np


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train.shape
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

y_train = np.eye(10)[y_train.astype("int")].reshape(50000, 10)
y_test = np.eye(10)[y_test.astype("int")].reshape(10000, 10)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=3, padding="same",
                 activation="relu", input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=3, padding="same",
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=2, padding="same",
                 activation="relu"))
model.add(Conv2D(filters=64, kernel_size=2, padding="same",
                 activation="relu"))
model.add(Conv2D(filters=32, kernel_size=2, padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=64,
                    nb_epoch=2,
                    verbose=1,
                    validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)
plot_model(model, show_shapes=True, to_file="data/model.png")
