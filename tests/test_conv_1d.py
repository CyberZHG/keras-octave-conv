import os
import tempfile
from unittest import TestCase
import numpy as np
from keras.layers import Input, MaxPool1D, Flatten, Dense
from keras.models import Model, load_model
from keras_octave_conv import OctaveConv1D, octave_dual


class TestConv1D(TestCase):

    def _test_fit(self, model):
        data_size = 4096
        x = np.random.standard_normal((data_size, 32, 3))
        y = np.random.randint(0, 1, data_size)
        model.fit(x, y, epochs=3)
        model_path = os.path.join(tempfile.gettempdir(), 'test_octave_conv_%f.h5' % np.random.random())
        model.save(model_path)
        model = load_model(model_path, custom_objects={'OctaveConv1D': OctaveConv1D})
        predicted = model.predict(x).argmax(axis=-1)
        diff = np.sum(np.abs(y - predicted))
        self.assertLess(diff, 100)

    def test_fit_default(self):
        inputs = Input(shape=(32, 3))
        high, low = OctaveConv1D(13, kernel_size=3)(inputs)
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        high, low = OctaveConv1D(7, kernel_size=3)([high, low])
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        conv = OctaveConv1D(5, kernel_size=3, ratio_out=0.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_octave(self):
        inputs = Input(shape=(32, 3))
        high, low = OctaveConv1D(13, kernel_size=3, octave=4)(inputs)
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        conv = OctaveConv1D(5, kernel_size=3, octave=4, ratio_out=0.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_fit_lower_output(self):
        inputs = Input(shape=(32, 3))
        high, low = OctaveConv1D(13, kernel_size=3)(inputs)
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        high, low = OctaveConv1D(7, kernel_size=3)([high, low])
        high, low = MaxPool1D()(high), MaxPool1D()(low)
        conv = OctaveConv1D(5, kernel_size=3, ratio_out=1.0)([high, low])
        flatten = Flatten()(conv)
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)

    def test_raise_dimension_specified(self):
        with self.assertRaises(ValueError):
            inputs = Input(shape=(32, None))
            outputs = OctaveConv1D(13, kernel_size=3, ratio_out=0.0)(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        with self.assertRaises(ValueError):
            inputs_high = Input(shape=(32, 3))
            inputs_low = Input(shape=(32, None))
            outputs = OctaveConv1D(13, kernel_size=3, ratio_out=0.0)([inputs_high, inputs_low])
            model = Model(inputs=[inputs_high, inputs_low], outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_raise_octave_divisible(self):
        with self.assertRaises(ValueError):
            inputs = Input(shape=(32, 3))
            outputs = OctaveConv1D(13, kernel_size=3, octave=5, ratio_out=0.0)(inputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def test_make_dual_layer(self):
        inputs = Input(shape=(32, 3))
        conv = OctaveConv1D(13, kernel_size=3)(inputs)
        pool = octave_dual(conv, MaxPool1D())
        conv = OctaveConv1D(7, kernel_size=3)(pool)
        pool = octave_dual(conv, MaxPool1D())
        conv = OctaveConv1D(5, kernel_size=3, ratio_out=0.0)(pool)
        flatten = octave_dual(conv, Flatten())
        outputs = Dense(units=2, activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary(line_length=200)
        self._test_fit(model)
