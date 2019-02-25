import tensorflow as tf


class Layers(object):
    @staticmethod
    def conv2d(inputs, in_channels, out_channels, kernel, strides, padding, name):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(scope.name + '_w',
                                      [kernel, kernel, in_channels, out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable(scope.name + '_b',
                                     [out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            conv = tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding=padding)
            pre_activation = tf.nn.bias_add(conv, biases)
            relu_activation = tf.nn.relu(pre_activation, name=scope.name)

        return relu_activation

    @staticmethod
    def atrous_conv2d(inputs, in_channels, out_channels, kernel, rate, padding, name):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(scope.name + '_w',
                                      [kernel, kernel, in_channels, out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable(scope.name + '_b',
                                     [out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            atrous_conv = tf.nn.atrous_conv2d(inputs, weights, rate, padding)
            pre_activation = tf.nn.bias_add(atrous_conv, biases)
            relu_activation = tf.nn.relu(pre_activation, name=scope.name)

        return relu_activation

    @staticmethod
    def max_pool2d(inputs, kernel, strides, padding, name):
        with tf.name_scope(name) as scope:
            pooled = tf.nn.max_pool(inputs, kernel, [1, strides, strides, 1], padding='VALID')

        return pooled

    @staticmethod
    def pad2d(inputs, pad=(0, 0), mode='CONSTANT', data_format='NHWC', scope=None):
        with tf.name_scope(scope, 'pad2d', [inputs]):
            if data_format == 'NHWC':
                paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
            elif data_format == 'NCHW':
                paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]

            net = tf.pad(inputs, paddings, mode=mode)

        return net
