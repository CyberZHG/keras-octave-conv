import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_octave_conv.backend import keras
from keras_octave_conv import OctaveConv2D, octave_conv_2d, octave_dual


class TestConv2D(TestCase):

    def _test_fit(self, model, data_format='channels_last'):
        data_size = 4096
        if data_format == 'channels_last':
            x = np.random.standard_normal((data_size, 32, 32, 3))
        else:
            x = np.random.standard_normal((data_size, 3, 32, 32))
        y = np.random.randint(0, 1, data_size)
        model.fit(x, y, epochs=3)
        model_path = os.path.join(tempfile.gettempdir(), 'test_octave_conv_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'OctaveConv2D': OctaveConv2D})
        predicted = model.predict(x).argmax(axis=-1)
        diff = np.sum(np.abs(y - predicted))
        self.assertLess(diff, 100)

    def test_fit_default(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3)(inputs)
        high, low = keras.layers.MaxPool2D()(high), keras.layers.MaxPool2D()(low)
        high, low = OctaveConv2D(7, kernel_size=3)([high, low])
        high, low = keras.layers.MaxPool2D()(high), keras.layers.MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0)([high, low])
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_channels_first(self):
        return 'The test needs GPU support'
        inputs = keras.layers.Input(shape=(3, 32, 32))
        high, low = OctaveConv2D(13, kernel_size=3, data_format='channels_first')(inputs)
        high = keras.layers.MaxPool2D(data_format='channels_first')(high)
        low = keras.layers.MaxPool2D(data_format='channels_first')(low)
        high, low = OctaveConv2D(7, kernel_size=3, data_format='channels_first')([high, low])
        high = keras.layers.MaxPool2D(data_format='channels_first')(high)
        low = keras.layers.MaxPool2D(data_format='channels_first')(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0, data_format='channels_first')([high, low])
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model, data_format='channels_first')

    def test_fit_octave(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3, octave=4)(inputs)
        high, low = keras.layers.MaxPool2D()(high), keras.layers.MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, octave=4, ratio_out=0.0)([high, low])
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_lower_output(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3)(inputs)
        high, low = keras.layers.MaxPool2D()(high), keras.layers.MaxPool2D()(low)
        high, low = OctaveConv2D(7, kernel_size=3)([high, low])
        high, low = keras.layers.MaxPool2D()(high), keras.layers.MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=1.0)([high, low])
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_raise_dimension_specified(self):
        with self.assertRaises(ValueError):
            inputs = keras.layers.Input(shape=(32, 32, None))
            outputs = OctaveConv2D(13, kernel_size=3, ratio_out=0.0)(inputs)
            model = keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        with self.assertRaises(ValueError):
            inputs_high = keras.layers.Input(shape=(32, 32, 3))
            inputs_low = keras.layers.Input(shape=(32, 32, None))
            outputs = OctaveConv2D(13, kernel_size=3, ratio_out=0.0)([inputs_high, inputs_low])
            model = keras.models.Model(inputs=[inputs_high, inputs_low], outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_raise_octave_divisible(self):
        with self.assertRaises(ValueError):
            inputs = keras.layers.Input(shape=(32, 32, 3))
            outputs = OctaveConv2D(13, kernel_size=3, octave=5, ratio_out=0.0)(inputs)
            model = keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_make_dual_lambda(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        conv = OctaveConv2D(13, kernel_size=3)(inputs)
        pool = octave_dual(conv, lambda: keras.layers.MaxPool2D())
        conv = OctaveConv2D(7, kernel_size=3)(pool)
        pool = octave_dual(conv, lambda: keras.layers.MaxPool2D())
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0)(pool)
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_octave_conv_high(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        conv = octave_conv_2d(inputs, filters=13, kernel_size=3)
        pool = octave_dual(conv, keras.layers.MaxPool2D())
        conv = octave_conv_2d(pool, filters=7, kernel_size=3, name='Octave-Mid')
        pool = octave_dual(conv, keras.layers.MaxPool2D())
        conv = octave_conv_2d(pool, filters=5, kernel_size=3, ratio_out=0.0)
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_octave_conv_low(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        conv = octave_conv_2d(inputs, filters=13, kernel_size=3)
        pool = octave_dual(conv, keras.layers.MaxPool2D())
        conv = octave_conv_2d(pool, filters=7, kernel_size=3, name='Octave-Mid')
        pool = octave_dual(conv, keras.layers.MaxPool2D())
        conv = octave_conv_2d(pool, filters=5, kernel_size=3, ratio_out=1.0)
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_stride(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        high, low = OctaveConv2D(13, kernel_size=3, strides=(1, 2))(inputs)
        high, low = keras.layers.MaxPool2D()(high), keras.layers.MaxPool2D()(low)
        conv = OctaveConv2D(5, kernel_size=3, ratio_out=0.0)([high, low])
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_octave_conv_stride(self):
        inputs = keras.layers.Input(shape=(32, 32, 3))
        conv = octave_conv_2d(inputs, filters=13, kernel_size=3, strides=(1, 2))
        pool = octave_dual(conv, keras.layers.MaxPool2D())
        conv = octave_conv_2d(pool, filters=5, kernel_size=3, ratio_out=1.0)
        flatten = keras.layers.Flatten()(conv)
        outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)
