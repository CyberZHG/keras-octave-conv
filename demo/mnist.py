import keras
import keras.backend as K
import numpy as np
from keras.layers import Input, BatchNormalization, MaxPool2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Model
from keras.datasets import fashion_mnist
from keras_octave_conv import OctaveConv2D


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.expand_dims(x_train.astype(K.floatx()) / 255, axis=-1)
x_test = np.expand_dims(x_test.astype(K.floatx()) / 255, axis=-1)

y_train, y_test = np.expand_dims(y_train, axis=-1), np.expand_dims(y_test, axis=-1)

train_num = round(x_train.shape[0] * 0.9)
x_train, x_valid = x_train[:train_num, ...], x_train[train_num:, ...]
y_train, y_valid = y_train[:train_num, ...], y_train[train_num:, ...]


# Octave Conv
inputs = Input(shape=(28, 28, 1))
normal = BatchNormalization()(inputs)
high, low = OctaveConv2D(64, kernel_size=3)(normal)
high, low = MaxPool2D()(high), MaxPool2D()(low)
high, low = OctaveConv2D(32, kernel_size=3)([high, low])
conv = OctaveConv2D(16, kernel_size=3, ratio_out=0.0)([high, low])
pool = MaxPool2D()(conv)
flatten = Flatten()(pool)
normal = BatchNormalization()(flatten)
dropout = Dropout(rate=0.4)(normal)
outputs = Dense(units=10, activation='softmax')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)]
)
octave_score = model.evaluate(x_test, y_test)
print('Accuracy of Octave: %.4f' % octave_score[1])


# Normal Conv
inputs = Input(shape=(28, 28, 1))
normal = BatchNormalization()(inputs)
conv = Conv2D(64, kernel_size=3, padding='same')(normal)
pool = MaxPool2D()(conv)
conv = Conv2D(32, kernel_size=3, padding='same')(pool)
conv = Conv2D(16, kernel_size=3, padding='same')(conv)
pool = MaxPool2D()(conv)
flatten = Flatten()(pool)
normal = BatchNormalization()(flatten)
dropout = Dropout(rate=0.4)(normal)
outputs = Dense(units=10, activation='softmax')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)]
)
normal_score = model.evaluate(x_test, y_test)
print('Accuracy of Octave: %.4f' % octave_score[1])
print('Accuracy of normal: %.4f' % normal_score[1])
