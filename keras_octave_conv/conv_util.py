import copy
from .backend import keras

__all__ = ['octave_dual']


def octave_dual(layers, builder):
    """Apply layers for outputs of octave convolution.

    :param layers: The outputs of octave convolution.
    :param builder: A function that builds the layer or just a layer.
    :return: The output tensors.
    """
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    if isinstance(builder, keras.layers.Layer):
        intermediates = [builder] + [copy.copy(builder) for _ in range(len(layers) - 1)]
    else:
        intermediates = [builder() for _ in range(len(layers))]
    for i, name in enumerate(['H', 'L']):
        if i < len(intermediates):
            try:
                intermediates[i].name += '-' + name
            except AttributeError as e:
                config = intermediates[i].get_config()
                config['name'] += '-' + name
                re_spawn_layer = intermediates[i].__class__.from_config(config)
                re_spawn_layer.set_weights(intermediates[i].get_weights())
                intermediates[i] = re_spawn_layer
    outputs = [intermediate(layers[i]) for i, intermediate in enumerate(intermediates)]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
