import tensorflow as tf
from Layers import convolution, down_convolution, up_convolution, get_num_channels

def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, n_channels, n_channels])
            if i == num_convolutions - 1:
                x = x + layer_input
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x

def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    n_channels = get_num_channels(layer_input)
    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            x = convolution(x, [5, 5, n_channels * 2, n_channels])
            x = x + layer_input
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
        return x

    with tf.variable_scope('conv_' + str(1)):
        x = convolution(x, [5, 5, n_channels * 2, n_channels])
        x = activation_fn(x)
        x = tf.nn.dropout(x, keep_prob)

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, n_channels, n_channels])
            if i == num_convolutions - 1:
                x = x + layer_input
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)

    return x


class VNet2D(object):
    def __init__(self,
                 num_classes=1,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation_fn=tf.nn.relu):
        """
        Implements VNet architecture https://arxiv.org/abs/1606.04797
        :param num_classes: Number of output classes.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation_fn


    def network_fn(self, x, keep_prob):
        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
        # channel

        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope('vnet/input_layer'):
            if input_channels == 1:
                x = tf.tile(x, [1, 1, 1, self.num_channels])
            else:
                x = self.activation_fn(convolution(x, [5, 5, input_channels, self.num_channels]))

        features = list()

        for l in range(self.num_levels):
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                x = convolution_block(x, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn)
                features.append(x)
                with tf.variable_scope('down_convolution'):
                    x = self.activation_fn(down_convolution(x, factor=2, kernel_size=[2, 2]))

        with tf.variable_scope('vnet/bottom_level'):
            x = convolution_block(x, self.bottom_convolutions, keep_prob, activation_fn=self.activation_fn)
        embedding = x
        for l in reversed(range(self.num_levels)):
            with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
                f = features[l]
                with tf.variable_scope('up_convolution'):
                    x = self.activation_fn(up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2]))

                x = convolution_block_2(x, f, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn)

        with tf.variable_scope('vnet/output_layer'):
            logits = convolution(x, [1, 1, self.num_channels, self.num_classes])

        return logits 
