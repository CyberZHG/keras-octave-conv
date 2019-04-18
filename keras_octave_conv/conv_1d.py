from keras.layers import Layer, Conv1D, AveragePooling1D, UpSampling1D
from keras import activations, initializers, regularizers, constraints
import keras.backend as K

__all__ = ['OctaveConv1D']


class OctaveConv1D(Layer):
    """Octave convolutions.

    # Arguments
        octave: The division of the spatial dimensions by a power of 2.
        ratio_out: The ratio of filters for lower spatial resolution.

    # References
        - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution]
          (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 octave=2,
                 ratio_out=0.5,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(OctaveConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.octave = octave
        self.ratio_out = ratio_out
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.filters_low = int(filters * self.ratio_out)
        self.filters_high = filters - self.filters_low

        self.conv_high_to_high, self.conv_low_to_high = None, None
        if self.filters_high > 0:
            self.conv_high_to_high = self._init_conv(self.filters_high, name='{}-Conv1D-HH'.format(self.name))
            self.conv_low_to_high = self._init_conv(self.filters_high, name='{}-Conv1D-LH'.format(self.name))
        self.conv_low_to_low, self.conv_high_to_low = None, None
        if self.filters_low > 0:
            self.conv_low_to_low = self._init_conv(self.filters_low, name='{}-Conv1D-HL'.format(self.name))
            self.conv_high_to_low = self._init_conv(self.filters_low, name='{}-Conv1D-LL'.format(self.name))
        self.pooling = AveragePooling1D(
            pool_size=self.octave,
            padding='valid',
            name='{}-AveragePooling1D'.format(self.name),
        )
        self.up_sampling = UpSampling1D(
            size=self.octave,
            name='{}-UpSampling1D'.format(self.name),
        )

    def _init_conv(self, filters, name):
        return Conv1D(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=name,
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if input_shape_high[-1] is None:
            raise ValueError('The channel dimension of the higher spatial inputs '
                             'should be defined. Found `None`.')
        if input_shape_low is not None and input_shape_low[-1] is None:
            raise ValueError('The channel dimension of the lower spatial inputs '
                             'should be defined. Found `None`.')
        if input_shape_high[-2] is not None and input_shape_high[-2] % self.octave != 0:
            raise ValueError('The length of the higher spatial inputs should be divisible by the octave. '
                             'Found {} and {}.'.format(input_shape_high, self.octave))
        if input_shape_low is None:
            self.conv_low_to_high, self.conv_low_to_low = None, None

        if self.conv_high_to_high is not None:
            with K.name_scope(self.conv_high_to_high.name):
                self.conv_high_to_high.build(input_shape_high)
        if self.conv_low_to_high is not None:
            with K.name_scope(self.conv_low_to_high.name):
                self.conv_low_to_high.build(input_shape_low)
        if self.conv_high_to_low is not None:
            with K.name_scope(self.conv_high_to_low.name):
                self.conv_high_to_low.build(input_shape_high)
        if self.conv_low_to_low is not None:
            with K.name_scope(self.conv_low_to_low.name):
                self.conv_low_to_low.build(input_shape_low)
        super(OctaveConv1D, self).build(input_shape)

    @property
    def trainable_weights(self):
        weights = []
        if self.conv_high_to_high is not None:
            weights += self.conv_high_to_high.trainable_weights
        if self.conv_low_to_high is not None:
            weights += self.conv_low_to_high.trainable_weights
        if self.conv_high_to_low is not None:
            weights += self.conv_high_to_low.trainable_weights
        if self.conv_low_to_low is not None:
            weights += self.conv_low_to_low.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        if self.conv_high_to_high is not None:
            weights += self.conv_high_to_high.non_trainable_weights
        if self.conv_low_to_high is not None:
            weights += self.conv_low_to_high.non_trainable_weights
        if self.conv_high_to_low is not None:
            weights += self.conv_high_to_low.non_trainable_weights
        if self.conv_low_to_low is not None:
            weights += self.conv_low_to_low.non_trainable_weights
        return weights

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None

        output_shape_high = None
        if self.filters_high > 0:
            output_shape_high = self.conv_high_to_high.compute_output_shape(input_shape_high)
        output_shape_low = None
        if self.filters_low > 0:
            output_shape_low = self.conv_high_to_low.compute_output_shape(
                self.pooling.compute_output_shape(input_shape_high),
            )

        if self.filters_low == 0:
            return output_shape_high
        if self.filters_high == 0:
            return output_shape_low
        return [output_shape_high, output_shape_low]

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs_high, inputs_low = inputs
        else:
            inputs_high, inputs_low = inputs, None

        outputs_high_to_high, outputs_low_to_high = 0.0, 0.0
        if self.conv_high_to_high is not None:
            outputs_high_to_high = self.conv_high_to_high(inputs_high)
        if self.conv_low_to_high is not None:
            outputs_low_to_high = self.up_sampling(self.conv_low_to_high(inputs_low))
        outputs_high = outputs_high_to_high + outputs_low_to_high

        outputs_low_to_low, outputs_high_to_low = 0.0, 0.0
        if self.conv_low_to_low is not None:
            outputs_low_to_low = self.conv_low_to_low(inputs_low)
        if self.conv_high_to_low is not None:
            outputs_high_to_low = self.conv_high_to_low(self.pooling(inputs_high))
        outputs_low = outputs_low_to_low + outputs_high_to_low

        if self.filters_low == 0:
            return outputs_high
        if self.filters_high == 0:
            return outputs_low
        return [outputs_high, outputs_low]

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'octave': self.octave,
            'ratio_out': self.ratio_out,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(OctaveConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
