# Keras Octave Conv

[![Travis](https://travis-ci.org/CyberZHG/keras-octave-conv.svg)](https://travis-ci.org/CyberZHG/keras-octave-conv)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-octave-conv/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-octave-conv)
![](https://img.shields.io/badge/license-MIT-blue.svg)
[![996.ICU](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://996.icu) 

Unofficial implementation of [Drop an Octave: Reducing Spatial Redundancy in
Convolutional Neural Networks with Octave Convolution](https://arxiv.org/pdf/1904.05049.pdf).

## Install

```bash
pip install keras-octave-conv
```

## Usage

The `OctaveConv2D` layer could be used just like the `Conv2D` layer, except the `padding` argument is forced to be `'same'`.

### First Octave

Use a single input for the first octave layer:

```python
from keras.layers import Input
from keras_octave_conv import OctaveConv2D

inputs = Input(shape=(32, 32, 3))
high, low = OctaveConv2D(filters=16, kernel_size=3, octave=2, ratio_out=0.125)(inputs)
```

The two outputs represent the results in higher and lower spatial resolutions.

Special arguments:
* `octave`: default is `2`. The division of the spatial dimensions.
* `ratio_out`: default is `0.5`. The ratio of filters for lower spatial resolution.

### Intermediate Octave

The intermediate octave layers takes two inputs and produce two outputs:

 ```python
from keras.layers import Input, MaxPool2D
from keras_octave_conv import OctaveConv2D

inputs = Input(shape=(32, 32, 3))
high, low = OctaveConv2D(filters=16, kernel_size=3)(inputs)

high, low = MaxPool2D()(high), MaxPool2D()(low)
high, low = OctaveConv2D(filters=8, kernel_size=3)([high, low])
```

Note that the same `octave` value should be used throughout the whole model.

### Last Octave

Set `ratio_out` to `0.0` to get a single output for further processing:

```python
from keras.layers import Input, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras_octave_conv import OctaveConv2D

inputs = Input(shape=(32, 32, 3))
high, low = OctaveConv2D(filters=16, kernel_size=3)(inputs)

high, low = MaxPool2D()(high), MaxPool2D()(low)
high, low = OctaveConv2D(filters=8, kernel_size=3)([high, low])

high, low = MaxPool2D()(high), MaxPool2D()(low)
conv = OctaveConv2D(filters=4, kernel_size=3, ratio_out=0.0)([high, low])

flatten = Flatten()(conv)
outputs = Dense(units=10, activation='softmax')(flatten)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
```

### Utility

`octave_dual` helps to create dual layers for processing the outputs of octave convolutions:

```python
from keras.layers import Input, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras_octave_conv import OctaveConv2D, octave_dual

inputs = Input(shape=(32, 32, 3))
conv = OctaveConv2D(filters=16, kernel_size=3)(inputs)

pool = octave_dual(conv, MaxPool2D())
conv = OctaveConv2D(filters=8, kernel_size=3)(pool)

pool = octave_dual(conv, MaxPool2D())
conv = OctaveConv2D(filters=4, kernel_size=3, ratio_out=0.0)(pool)

flatten = Flatten()(conv)
outputs = Dense(units=10, activation='softmax')(flatten)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
```

`octave_conv_2d` creates the octave structure with built-in Keras layers:

```python
from keras.layers import Input, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras.utils import plot_model
from keras_octave_conv import octave_conv_2d, octave_dual

inputs = Input(shape=(32, 32, 3), name='Input')
conv = octave_conv_2d(inputs, filters=16, kernel_size=3, name='Octave-First')

pool = octave_dual(conv, MaxPool2D(name='Pool-1'))
conv = octave_conv_2d(pool, filters=8, kernel_size=3, name='Octave-Mid')

pool = octave_dual(conv, MaxPool2D(name='Pool-2'))
conv = octave_conv_2d(pool, filters=4, kernel_size=3, ratio_out=0.0, name='Octave-Last')

flatten = Flatten(name='Flatten')(conv)
outputs = Dense(units=10, activation='softmax', name='Output')(flatten)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
plot_model(model, to_file='octave_model.png')
```

![](./octave_model.png)
