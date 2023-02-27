import tensorflow as tf
import tensorflow.contrib.slim as slim


def my_vgg_arg_scope(weight_decay=0.0005,
                     use_batch_norm=False,
                     is_training=True,
                     batch_norm_decay=0.95,
                     use_scale=True):
  """ borrowed from EthanGao """
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = {'is_training': is_training,
                         'decay': batch_norm_decay,
                         'updates_collections': None, 'scale': use_scale}
  else:
    normalizer_fn = None
    normalizer_params = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
        dtype=tf.float32),
      biases_initializer=tf.constant_initializer(value=0.0, dtype=tf.float32),
      normalizer_fn=normalizer_fn,
      normalizer_params=normalizer_params):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        data_format='NHWC'):  # data_format='NCHW'
      with slim.arg_scope([slim.max_pool2d], padding='VALID',
                          data_format='NHWC'):
        with slim.arg_scope([slim.batch_norm], **normalizer_params) as arg_sc:
          return arg_sc


def grid_net_embed(x_img, add_on_img, enc_dim, spa_ch_dim, output_size=None):
    """
    :param x_img: [bs, 176, 200, 10]
    :param add_on_img: [bs, 176, 200, dim]
    :return:
    """
    def dilated_conv(x_enc, rates=(3, 6, 12, 24), c_dim=16, k_size=3):
      xx_rates = []
      for rate in rates:
        xx = slim.conv2d(x_enc, c_dim, [k_size, k_size], rate=rate,
                         scope=('rate%d_conv1' % rate))
        xx = slim.conv2d(xx, c_dim, [1, 1], scope=('rate%d_conv2' % rate))
        xx_rates.append(xx)
      x_out = tf.add_n(xx_rates, name='added_out')
      return x_out

    if output_size is None:
      output_size = [tf.shape(x_img)[1], tf.shape(x_img)[2]]
    with tf.variable_scope('gridnet_enc'):
      with slim.arg_scope(my_vgg_arg_scope()):
        x = x_img  # NHWC
        x = slim.conv2d(x, enc_dim, [1, 1], scope='x_img_fc1')
        x = slim.conv2d(x, enc_dim, [1, 1], scope='x_img_fc2')
        x = tf.concat([x, add_on_img], axis=-1)  # [176, 200, dim + dim]
        x = slim.conv2d(x, spa_ch_dim, [1, 1], scope='x_img_fc3')

        x_ori = x
        with tf.variable_scope('large'):
          x_l = slim.conv2d(x_ori, int(spa_ch_dim/2), [3, 3], scope='conv1')
          x_l = slim.conv2d(x_l, int(spa_ch_dim/2), [3, 3], scope='conv2')
          x_l = dilated_conv(x_l, c_dim=int(spa_ch_dim/2), k_size=3)
          x_l = slim.conv2d(x_l, int(spa_ch_dim/2), [3, 3], scope='conv3')
          # resize
          x_l_resize = tf.image.resize_bilinear(x_l, output_size, name='resize')

        with tf.variable_scope('medium'):
          x_m = slim.conv2d(x_ori, int(spa_ch_dim/2), [3, 3], scope='conv1')
          x_m = slim.max_pool2d(x_m, [2, 2], scope='pool1')  # [88, 100]
          x_m = slim.conv2d(x_m, spa_ch_dim, [3, 3], scope='conv2')
          x_m = slim.max_pool2d(x_m, [2, 2], scope='pool2')  # [44, 50]
          x_m = dilated_conv(x_m, c_dim=spa_ch_dim, k_size=3)
          x_m = slim.conv2d(x_m, spa_ch_dim, [3, 3], scope='conv3')
          # resize
          x_m_resize = tf.image.resize_bilinear(x_m, output_size, name='resize')

        with tf.variable_scope('small'):
          x_s = slim.conv2d(x_ori, int(spa_ch_dim/2), [3, 3], scope='conv1')
          x_s = slim.max_pool2d(x_s, [2, 2], scope='pool1')  # [88, 100]
          x_s = slim.conv2d(x_s, spa_ch_dim, [3, 3], scope='conv2')
          x_s = slim.max_pool2d(x_s, [2, 2], scope='pool2')  # [44, 50]
          x_s = slim.conv2d(x_s, spa_ch_dim, [3, 3], scope='conv3')
          x_s = slim.max_pool2d(x_s, [2, 2], scope='pool3')  # [22, 25]
          x_s = slim.conv2d(x_s, spa_ch_dim*2, [3, 3], scope='conv4')
          x_s = slim.max_pool2d(x_s, [2, 2], scope='pool4')  # [11, 13]
          x_s = dilated_conv(x_s, rates=(2, 4, 6),
                             c_dim=spa_ch_dim*2, k_size=3)
          x_s = slim.conv2d(x_s, spa_ch_dim*2, [3, 3], scope='conv5')
          # resize
          x_s_resize = tf.image.resize_bilinear(x_s, output_size, name='resize')

        spa_embed = tf.concat([x_l_resize, x_m_resize, x_s_resize], axis=-1)
        # spa_embed = slim.conv2d(spa_embed, enc_dim, [1, 1], scope='spa_fc')

        return spa_embed, x_s
