from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import numpy as np


CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 64


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

mean = np.mean(X_train, axis=(0, 1, 2, 3))
std = np.std(X_train, axis=(0, 1, 2, 3))
X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

y_train = to_categorical(y_train, num_classes=CLASSES)
y_test = to_categorical(y_test, num_classes=CLASSES)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=5, strides=2,
                 activation="relu", input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=3, padding="same",
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=3, padding="same",
                 activation="relu"))
model.add(Conv2D(filters=512, kernel_size=3, padding="same",
                 activation="relu"))
model.add(Conv2D(filters=256, kernel_size=3, padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(decay=1e-4)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.summary()

datagen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)

# model.fit(X_train, y_train,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCHS,
#           verbose=1,
#           validation_split=0.1)

model.fit_generator(datagen.flow(X_train, y_train, BATCH_SIZE),
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    epochs=EPOCHS)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)
