import copy
from keras.layers import Layer

__all__ = ['octave_dual']


def octave_dual(layers, builder):
    """Apply layers for outputs of octave convolution.

    :param layers: The outputs of octave convolution.
    :param builder: A function that builds the layer or just a layer.
    :return: The output tensors.
    """
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    if isinstance(builder, Layer):
        intermediates = [builder] + [copy.copy(builder) for _ in range(len(layers) - 1)]
    else:
        intermediates = [builder() for _ in range(len(layers))]
    for i, name in enumerate(['H', 'L']):
        if i < len(intermediates):
            intermediates[i].name += '-' + name
    outputs = [intermediate(layers[i]) for i, intermediate in enumerate(intermediates)]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
