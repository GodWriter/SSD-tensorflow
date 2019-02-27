import tensorflow as tf

from collections import namedtuple
from layers import Layers


SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """
    The default features layers with 300*300 image input are:
    conv4 ==> 38 x 28
    conv7 ==> 19 x 19
    conv8 ==> 10 x 10
    conv9 ==> 5 x 5
    conv10 ==> 3 x 3
    conv11 ==> 1 x 1
    The default image size used to train this network is 300*300
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, args, params=None):
        self.args = args

        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    def net(self, inputs,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='ssd_300_vgg'):

        r = ssd_net(self.args,
                    inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse,
                    scope=scope
                    )

        pass


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def ssd_multibox_layer(net,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    if normalization > 0:
        net = Layers.l2_normalization(net, scaling=True)

    # Number of anchors
    num_anchors = len(sizes) + len(ratios)

    # Location
    num_loc_pred = num_anchors * 4
    loc_pred = Layers.conv2d(net, net.get_shape[-1], num_loc_pred, 3, 1, 'SAME', 'conv_loc', activation_fn=False)
    loc_pred = Layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

    # Class prediction
    pass




def ssd_net(args,
            inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='ssd_300_vgg'):
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, default_name='ssd_300_vgg', values=[inputs], reuse=reuse):
        # Original VGG-16 blocks.
        # Block 1
        net = Layers.conv2d(inputs, 3, 64, 3, 1, 'SAME', 'block1_conv1')
        net = Layers.conv2d(net, 64, 64, 3, 1, 'SAME', 'block1_conv2')
        end_points['block1'] = net
        net = Layers.max_pool2d(net, 2, 2, 'VALID', 'block1_pool')
        # Block 2
        net = Layers.conv2d(net, 64, 128, 3, 1, 'SAME', 'block2_conv1')
        net = Layers.conv2d(net, 128, 128, 3, 1, 'SAME', 'block2_conv2')
        end_points['block2'] = net
        net = Layers.max_pool2d(net, 2, 2, 'VALID', 'block2_pool')
        # Block 3
        net = Layers.conv2d(net, 128, 256, 3, 1, 'SAME', 'block3_conv1')
        net = Layers.conv2d(net, 256, 256, 3, 1, 'SAME', 'block3_conv2')
        net = Layers.conv2d(net, 256, 256, 3, 1, 'SAME', 'block3_conv3')
        end_points['block3'] = net
        net = Layers.max_pool2d(net, 2, 2, 'VALID', 'block3_pool')
        # Block 4
        net = Layers.conv2d(net, 256, 512, 3, 1, 'SAME', 'block4_conv1')
        net = Layers.conv2d(net, 512, 512, 3, 1, 'SAME', 'block4_conv2')
        net = Layers.conv2d(net, 512, 512, 3, 1, 'SAME', 'block4_conv3')
        end_points['block4'] = net
        net = Layers.max_pool2d(net, 2, 2, 'VALID', 'block4_pool')
        # Block 5
        net = Layers.conv2d(net, 512, 512, 3, 1, 'SAME', 'block5_conv1')
        net = Layers.conv2d(net, 512, 512, 3, 1, 'SAME', 'block5_conv2')
        net = Layers.conv2d(net, 512, 512, 3, 1, 'SAME', 'block5_conv3')
        end_points['block5'] = net
        net = Layers.max_pool2d(net, 2, 2, 'VALID', 'block5_pool')

        # Additional SSD blocks
        # Block 6
        net = Layers.atrous_conv2d(net, 512, 1024, 3, 6, 'SAME', scope='block6_atrous_conv')
        end_points['block6'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=args.is_training)
        # Block 7
        net = Layers.conv2d(net, 1024, 1024, 1, 1, 'SAME', 'block7_conv')
        end_points['block7'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=args.is_training)

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except last 2)
        # Block 8
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = Layers.conv2d(net, 1024, 256, 1, 1, 'SAME', 'conv1x1')
            net = Layers.pad2d(net, pad=(1, 1))
            net = Layers.conv2d(net, 256, 512, 3, 2, 'VALID', 'conv3x3')
        end_points[end_point] = net
        # Block 9
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = Layers.conv2d(net, 512, 128, 1, 1, 'SAME', 'conv1x1')
            net = Layers.pad2d(net, pad=(1, 1))
            net = Layers.conv2d(net, 128, 256, 3, 2, 'VALID', 'conv3x3')
        end_points[end_point] = net
        # Block 10
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = Layers.conv2d(net, 256, 128, 1, 1, 'SAME', 'conv1x1')
            net = Layers.pad2d(net, pad=(1, 1))
            net = Layers.conv2d(net, 128, 256, 3, 1, 'VALID', 'conv3x3')
        end_points[end_point] = net
        # Block 11
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = Layers.conv2d(net, 256, 128, 1, 1, 'SAME', 'conv1x1')
            net = Layers.conv2d(net, 128, 256, 3, 1, 'VALID', 'conv3x3')
        end_points[end_point] = net

        # Prediction and localisations layers
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                prediction_, localisation_ = ssd_multibox_layer(end_points[layer],
                                                                num_classes,
                                                                anchor_sizes[i],
                                                                anchor_ratios[i],
                                                                normalizations[i])
                pass
