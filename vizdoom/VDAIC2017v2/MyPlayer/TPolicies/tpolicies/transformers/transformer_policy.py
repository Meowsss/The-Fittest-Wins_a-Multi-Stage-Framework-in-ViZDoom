from tpolicies.base_policy import BasePolicy
from tpolicies.utils.distributions import CategoricalPdType
from gym import spaces
from gym.spaces import Box, Tuple
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from collections import namedtuple
INF = 1e20


def ln(inputs, epsilon=1e-8, scope="ln"):
  '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
  inputs: A tensor with 2 or more dimensions, where the first dimension
  has `batch_size`.
  epsilon: A floating number. A very small number for preventing ZeroDivision
  Error.
  scope: Optional scope for `variable_scope`.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta", params_shape,
                           initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape,
                            initializer=tf.ones_initializer())
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

  return outputs


def trans_mask(inputs, queries=None, keys=None, type=None):
  """Masks paddings on keys or queries to inputs
  inputs: 3d tensor. (N, T_q, T_k)
  queries: 3d tensor. (N, T_q, d)
  keys: 3d tensor. (N, T_k, d)

  e.g.,
  >> queries = tf.constant([[[1.],
                      [2.],
                      [0.]]], tf.float32) # (1, 3, 1)
  >> keys = tf.constant([[[4.],
                   [0.]]], tf.float32)  # (1, 2, 1)
  >> inputs = tf.constant([[[4., 0.],
                             [8., 0.],
                             [0., 0.]]], tf.float32)
  >> mask(inputs, queries, keys, "key")
  array([[[ 4.0000000e+00, -4.2949673e+09],
      [ 8.0000000e+00, -4.2949673e+09],
      [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
  >> inputs = tf.constant([[[1., 0.],
                           [1., 0.],
                            [1., 0.]]], tf.float32)
  >> mask(inputs, queries, keys, "query")
  array([[[1., 0.],
      [1., 0.],
      [0., 0.]]], dtype=float32)
  """
  padding_num = -2 ** 32 + 1
  if type in ("k", "key", "keys"):
    # Generate masks
    masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
    masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
    masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

    # Apply masks to inputs
    paddings = tf.ones_like(inputs) * padding_num
    outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
  elif type in ("q", "query", "queries"):
    # Generate masks
    masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

    # Apply masks to inputs
    outputs = inputs * masks
  elif type in ("f", "future", "right"):
    diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
    tril = tf.linalg.LinearOperatorLowerTriangular(
      diag_vals).to_dense()  # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(tril, 0),
                    [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

    paddings = tf.ones_like(masks) * padding_num
    outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
  else:
    print("Check if you entered type correctly!")

  return outputs


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 pointer=False,
                                 scope="scaled_dot_product_attention"):
  '''See 3.2.1.
  Q: Packed queries. 3d tensor. [N, T_q, d_k].
  K: Packed keys. 3d tensor. [N, T_k, d_k].
  V: Packed values. 3d tensor. [N, T_k, d_v].
  causality: If True, applies masking for future blinding
  dropout_rate: A floating point number of [0, 1].
  training: boolean for controlling droput
  scope: Optional scope for `variable_scope`.
  '''
  with tf.variable_scope(scope):
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # key masking
    outputs = trans_mask(outputs, Q, K, type="key")

    # causality or future blinding masking
    if causality:
      outputs = trans_mask(outputs, type="future")

    # softmax
    output_logits = outputs
    outputs = tf.nn.softmax(outputs)
    if pointer:
      return output_logits, outputs
    attention = tf.transpose(outputs, [0, 2, 1])
    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    # query masking
    outputs = trans_mask(outputs, Q, K, type="query")

    # dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

  return outputs


def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0.0,
                        training=True,
                        causality=False,
                        pointer=False,
                        scope="multihead_attention"):
  '''Applies multihead attention. See 3.2.2
  queries: A 3d tensor with shape of [N, T_q, d_model].
  keys: A 3d tensor with shape of [N, T_k, d_model].
  values: A 3d tensor with shape of [N, T_k, d_model].
  num_heads: An int. Number of heads.
  dropout_rate: A floating point number.
  training: Boolean. Controller of mechanism for dropout.
  causality: Boolean. If true, units that reference the future are masked.
  scope: Optional scope for `variable_scope`.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  d_model = queries.get_shape().as_list()[-1]
  with tf.variable_scope(scope):
    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                   axis=0)  # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)

    # Attention
    if pointer:
      # do not use multi-head
      logits, pd = scaled_dot_product_attention(Q, K, V, causality,
                                                dropout_rate, training, pointer)
      return logits, pd
    else:
      outputs = scaled_dot_product_attention(Q_, K_, V_, causality,
                                             dropout_rate, training, pointer)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                        axis=2)  # (N, T_q, d_model)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = ln(outputs)

  return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
  '''position-wise feed forward net. See 3.3

  inputs: A 3d tensor with shape of [N, T, C].
  num_units: A list of two integers.
  scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope):
    # Inner layer
    outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

    # Outer layer
    outputs = tf.layers.dense(outputs, num_units[1])

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = ln(outputs)

  return outputs


def my_observation_input(ob_space, batch_size=None, name='Ob'):
  '''
  Build observation input with encoding depending on the
  observation space type
  Params:

  ob_space: observation space (should be one of gym.spaces)
  batch_size: batch size for input (default is None, so that resulting input
  placeholder can take tensors with any batch size)
  name: tensorflow variable name for input placeholder

  returns: tuple (input_placeholder, processed_input_tensor)
  '''
  if isinstance(ob_space, spaces.Discrete):
    input_x = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
    processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
    return input_x, processed_x

  elif isinstance(ob_space, spaces.Box):
    input_shape = (batch_size,) + ob_space.shape
    input_x = tf.placeholder(shape=input_shape, dtype=ob_space.dtype, name=name)
    processed_x = tf.to_float(input_x)
    return input_x, processed_x

  else:
    raise NotImplementedError


def gym_observation_input(input_data, ob_space, batch_size=None, name='Ob'):
    '''
    Build observation input with encoding depending on the
    observation space type
    Params:

    ob_space: observation space (should be one of gym.spaces)
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    '''
    if input_data is None:
        if isinstance(ob_space, Box):
            input_shape = (batch_size,) + ob_space.shape
            input_x = tf.placeholder(shape=input_shape, dtype=ob_space.dtype, name=name)
            processed_x = tf.to_float(input_x)
            return input_x, processed_x

        elif isinstance(ob_space, Tuple):
            assert all([isinstance(space, Box) for space in ob_space.spaces])
            input_x = tuple([tf.placeholder(shape=(batch_size,) + space.shape,
                                            dtype=space.dtype, name=name)
                             for space in ob_space.spaces])
            processed_x = tuple([tf.to_float(input) for input in input_x])
            return input_x, processed_x

        else:
            raise NotImplementedError
    else:
        if isinstance(ob_space, Box):
            return input_data.X, tf.to_float(input_data.X)

        elif isinstance(ob_space, Tuple):
            assert all([isinstance(space, Box) for space in ob_space.spaces])
            processed_x = [tf.to_float(x) for x in input_data.X] \
                if isinstance(input_data.X, tuple) else tf.to_float(input_data.X)
            return input_data.X, processed_x

        else:
            raise NotImplementedError


class Transformer(BasePolicy):
  def __init__(self, sess, ob_space, ac_space, nbatch=None, reuse=False,
               env='5I', input_data=None, test=False):
    self.test = test
    self.num_enc_blocks = 3
    self.num_dec_blocks = 3
    self.dropout_rate = 0.5
    self.num_act_heads = 8

    self.env = env
    assert isinstance(ob_space, spaces.Tuple)
    self.ob_space = ob_space
    assert isinstance(ac_space, spaces.Tuple)
    self.ac_space = ac_space

    for i in range(len(ac_space.spaces)):
      if i != len(ac_space.spaces) - 1:
        assert isinstance(ac_space.spaces[i], spaces.Discrete)
      else:
        assert isinstance(ac_space.spaces[i], spaces.Box)

    self.ability_dim = ac_space.spaces[0].n
    self.shift_dim = ac_space.spaces[1].n
    self.noop_num_dim = ac_space.spaces[2].n
    self.tar_unit_dim = ac_space.spaces[3].n
    self.tar_loc_x_dim = ac_space.spaces[4].n
    self.tar_loc_y_dim = ac_space.spaces[5].n
    self.select_dim = ac_space.spaces[6].n

    self.ff_dim = 64
    self.enc_dim = 128

    masks, labels = {}, {}
    if input_data is None:
      X_list, processed_x_list = my_observation_input(ob_space.spaces[0], nbatch)
      X_vec, processed_x_vec = my_observation_input(ob_space.spaces[1], nbatch)
      # X_img, processed_x_img = my_observation_input(ob_space.spaces[1], nbatch)

      masks['len'] = tf.placeholder(shape=(nbatch, self.select_dim),
                                    dtype=tf.float32, name='len_mask')
      masks['ability'] = tf.placeholder(shape=(nbatch, self.ability_dim),
                                        dtype=tf.float32, name='ability_mask')
      masks['head'] = tf.placeholder(shape=(nbatch, self.ability_dim, self.num_act_heads),
                                     dtype=tf.float32, name='head_mask')
      masks['select'] = tf.placeholder(shape=(nbatch, self.ability_dim, self.select_dim),
                                       dtype=tf.float32, name='select_mask')
      masks['target'] = tf.placeholder(shape=(nbatch, self.ability_dim, self.select_dim),
                                       dtype=tf.float32, name='target_mask')
      labels['ability'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                         name='ability_labels')
      labels['shift'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                       name='shift_labels')
      labels['noop_num'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='noop_num_labels')
      labels['tar_unit'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='tar_unit_labels')
      labels['tar_x'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                       name='tar_x_labels')
      labels['tar_y'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                       name='tar_y_labels')
      labels['s_select'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='s_select_labels')
      labels['m_select'] = tf.placeholder(shape=(nbatch, self.select_dim),
                                          dtype=tf.int32, name='m_select_labels')
    else:
      processed_x_list = input_data.X_UNITS
      X_list = processed_x_list
      processed_x_vec = input_data.X_GLOBAL
      X_vec = processed_x_vec

      masks['len'] = self._process_mask(input_data.LEN_MASKS)
      masks['ability'] = self._process_mask(input_data.ABILITY_MASKS)
      masks['head'] = self._process_mask(input_data.HEAD_MASKS)
      masks['select'] = self._process_mask(input_data.SELECT_MASKS)
      masks['target'] = self._process_mask(input_data.TARGET_MASKS)

      labels['ability'] = input_data.ABILITY_LABELS
      labels['shift'] = input_data.SHIFT_LABELS
      labels['noop_num'] = input_data.NOOP_NUM_LABELS
      labels['tar_unit'] = input_data.TAR_UNIT_LABELS
      labels['tar_x'] = input_data.TAR_X_LABELS
      labels['tar_y'] = input_data.TAR_Y_LABELS
      labels['s_select'] = input_data.S_SELECT_LABELS
      labels['m_select'] = input_data.M_SELECT_LABELS

    self.processed_x_vec = processed_x_vec

    with tf.variable_scope("model", reuse=reuse):
      rl_outputs, il_outputs = self._create_net(x=processed_x_list,
                                                masks=masks,
                                                labels=labels)
      mha_argmax_tf, mha_tf, neglogp_tf, vf_tf = rl_outputs
      il_loss_tf = il_outputs[0]
      losses_tf = il_outputs[1]

    self.initial_state = None
    self.il_loss = il_loss_tf
    self.losses = losses_tf

    def step(ob, *_args, **_kwargs):
      """ ob is a tuple of two: one is the state, one is the masks dict """
      state_list, state_vec, masks_feed = ob
      a, v, neglogp = sess.run([mha_tf, vf_tf, neglogp_tf],
                               {X_list: state_list,
                                X_vec: state_vec,
                                masks['len']: masks_feed['len'],
                                masks['select']: masks_feed['select'],
                                masks['target']: masks_feed['target'],
                                masks['head']: masks_feed['head'],
                                masks['ability']: masks_feed['ability']
                                })
      return a, v, self.initial_state, neglogp

    def value(ob, *_args, **_kwargs):
      state_list, state_vec, masks_feed = ob
      return sess.run(vf_tf,
                      {X_list: state_list,
                       X_vec: state_vec,
                       masks['len']: masks_feed['len'],
                       masks['select']: masks_feed['select'],
                       masks['target']: masks_feed['target'],
                       masks['head']: masks_feed['head'],
                       masks['ability']: masks_feed['ability']
                       })

    def act(ob):
      state_list, state_vec, masks_feed = ob
      return sess.run(mha_tf,
                      {X_list: state_list,
                       X_vec: state_vec,
                       masks['len']: masks_feed['len'],
                       masks['select']: masks_feed['select'],
                       masks['target']: masks_feed['target'],
                       masks['head']: masks_feed['head'],
                       masks['ability']: masks_feed['ability']
                       })

    # self.names = [X.name, head_a0.name, loc_a0.name, mov_seq_a0.name,
    # atk_seq_a0.name, vf.name, neglogp0.name]

    self.X_list = X_list
    self.X_vec = X_vec
    self.vf = vf_tf
    self.a_argmax = mha_argmax_tf
    self.a = mha_tf
    self.neglogp = neglogp_tf
    self.step = step
    self.value = value
    self.il_loss = il_loss_tf
    self.act = act

  def _create_net(self, x, masks, labels):
    self.head_pdtype = CategoricalPdType(ncat=self.ability_dim)
    self.loc_x_pdtype = CategoricalPdType(ncat=self.tar_loc_x_dim)
    self.loc_y_pdtype = CategoricalPdType(ncat=self.tar_loc_y_dim)
    self.ptr_pdtype = CategoricalPdType(ncat=self.tar_unit_dim)

    # x representation
    with tf.variable_scope("enc"):
      x = slim.fully_connected(x, self.enc_dim, scope='x_fc1')
      memory = self.encode(x, masks['len'], training=True)

    """ begin multi-head """
    # ability
    with tf.variable_scope("ability"):
      ability_logits, ability_argmax, ability_sam, ability_neglogp = \
        self._make_general_head(memory=memory,
                                n_head=self.ability_dim,
                                mask=masks['ability'])
    # shift
    with tf.variable_scope("shift"):
      shift_logits, shift_argmax, shift_sam, shift_neglogp = \
        self._make_general_head(memory=memory,
                                n_head=self.shift_dim)
    # noop_num
    with tf.variable_scope("noop_num"):
      noop_num_logits, noop_num_argmax, noop_num_sam, noop_num_neglogp = \
        self._make_general_head(memory=memory,
                                n_head=self.noop_num_dim)
    """ below requires the auto-regressive input """
    if not self.test:
      ability_taken_f = self._process_mask(tf.one_hot(labels['ability'],
                                                      depth=self.ability_dim))
    else:
      ability_taken_f = self._process_mask(tf.one_hot(ability_sam,
                                                      depth=self.ability_dim))
    ability_taken_f = tf.stop_gradient(ability_taken_f)
    # tar_unit
    with tf.variable_scope("tar_unit"):
      tar_unit_logits, tar_unit_argmax, tar_unit_sam, tar_unit_neglogp = \
        self._make_ptr_head(query=ability_taken_f,
                            ptr_mask=tf.reduce_sum(
                              tf.multiply(tf.expand_dims(ability_taken_f,
                                                         axis=-1),
                                          masks['target']),
                              axis=1),
                            memory=memory)
    # tar_loc_x
    with tf.variable_scope("tar_loc_x"):
      tar_loc_x_logits, tar_loc_x_argmax, tar_loc_x_sam, tar_loc_x_neglogp = \
        self._make_general_head_with_mask(memory=memory,
                                          mask=ability_taken_f,
                                          n_head=self.tar_loc_x_dim)
    # tar_loc_y
    with tf.variable_scope("tar_loc_y"):
      tar_loc_y_logits, tar_loc_y_argmax, tar_loc_y_sam, tar_loc_y_neglogp = \
        self._make_general_head_with_mask(memory=memory,
                                          mask=ability_taken_f,
                                          n_head=self.tar_loc_y_dim)
    # single_select
    with tf.variable_scope("select"):
      s_select_logits, s_select_argmax, s_select_sam, s_select_neglogp = \
        self._make_ptr_head(query=ability_taken_f,
                            ptr_mask=tf.reduce_sum(
                              tf.multiply(tf.expand_dims(ability_taken_f,
                                                         axis=-1),
                                          masks['select']),
                              axis=1),
                            memory=memory)
    # multi_select
    with tf.variable_scope("select"):
      m_select_logits, m_select_argmax, m_select_sam, m_select_neglogp = \
        self._make_multi_bi_head(memory=memory,
                                 select_mask=tf.reduce_sum(
                                   tf.multiply(tf.expand_dims(ability_taken_f,
                                                              axis=-1),
                                               masks['select']),
                                   axis=1)
                                 )

    with tf.variable_scope("RL"):
      neglogp_final = None
      vf = None
      # create final sampled actions
      actions_sam = [ability_sam, shift_sam, noop_num_sam, tar_unit_sam,
                     tar_loc_x_sam, tar_loc_y_sam, s_select_sam, m_select_sam]
      # create final neglogp
      # neglogp_final = ability_neglogp + shift_neglogp + noop_num_neglogp + \
      #                 tf.reduce_sum(
      #                   tf.multiply(masks['head'][:, -5:],
      #                               tf.concat([tar_unit_neglogp,
      #                                          tar_loc_x_neglogp,
      #                                          tar_loc_y_neglogp,
      #                                          s_select_neglogp,
      #                                          tf.reduce_sum(
      #                                            tf.multiply(masks['len'],
      #                                                        m_select_neglogp),
      #                                            axis=-1)],
      #                                         axis=-1)), axis=-1)
      # create value
      # memory_pooling = tf.layers.max_pooling1d(memory,
      #                                          pool_size=self.tar_unit_dim,
      #                                          strides=1)
      # vf = tf.reshape(memory_pooling, shape=[-1, self.enc_dim])
      # vf = slim.fully_connected(vf, 128, scope='v_fc1')
      # vf = slim.fully_connected(vf, 128, scope='v_fc2')
      # vf = slim.fully_connected(vf, 1, scope='v_out', activation_fn=None,
      #                           normalizer_fn=None)

    with tf.variable_scope("IL"):
      ability_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['ability'], logits=ability_logits)
      shift_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['shift'], logits=shift_logits)
      noop_num_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['noop_num'], logits=noop_num_logits)
      tar_unit_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['tar_unit'], logits=tar_unit_logits)
      tar_x_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['tar_x'], logits=tar_loc_x_logits)
      tar_y_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['tar_y'], logits=tar_loc_y_logits)
      s_select_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['s_select'], logits=s_select_logits)
      m_select_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels['m_select'], logits=m_select_logits)
      m_select_loss = tf.reduce_sum(m_select_loss, axis=-1)

      head_mask_selected = tf.reduce_sum(
        tf.multiply(tf.expand_dims(ability_taken_f, axis=-1),
                    masks['head']), axis=1)
      final_loss = ability_loss + shift_loss + noop_num_loss + \
         tf.reduce_sum(
           tf.multiply(
             tf.slice(head_mask_selected, begin=[0, 3], size=[-1, 5]),
             tf.concat([tf.expand_dims(tar_unit_loss, axis=-1),
                        tf.expand_dims(tar_x_loss, axis=-1),
                        tf.expand_dims(tar_y_loss, axis=-1),
                        tf.expand_dims(s_select_loss, axis=-1),
                        tf.expand_dims(m_select_loss, axis=-1)], axis=-1)
           ), axis=-1)
    final_loss = tf.reduce_mean(final_loss)
    losses = [ability_loss, shift_loss, noop_num_loss, tar_unit_loss,
              tar_x_loss, tar_y_loss, s_select_loss, m_select_loss]

    actions_argmax = [ability_argmax, shift_argmax, noop_num_argmax,
                      tar_unit_argmax,
                      tar_loc_x_argmax, tar_loc_y_argmax, s_select_argmax,
                      m_select_argmax]

    return [actions_argmax, actions_sam, neglogp_final, vf], \
           [final_loss, losses]

  def _make_general_head(self, memory, n_head, mask=None):
    """
    :param memory: [None, num_units, encoding dim]
    :param n_head: output dim
    :param pd_type:
    :return:
    """
    # feature-wise max-pooling [Relational Deep RL, deepmind]
    head_h = tf.layers.max_pooling1d(memory,
                                     pool_size=self.tar_unit_dim,
                                     strides=1)
    head_h = tf.reshape(head_h, shape=[-1, self.enc_dim])
    # concat with the global vector feature
    head_h = tf.concat([head_h, self.processed_x_vec], axis=-1)

    head_h = slim.fully_connected(head_h, 128, scope='fc1')
    head_h = slim.fully_connected(head_h, 128, scope='fc2')
    head_logits = slim.fully_connected(head_h,
                                       n_head,
                                       scope='logits',
                                       activation_fn=None,
                                       normalizer_fn=None)
    if mask is not None:
      head_logits = self._mask_logits(head_logits, mask)

    head_argmax = tf.argmax(head_logits, axis=-1)
    head_pd = self.head_pdtype.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    return head_logits, head_argmax, head_sam, head_neglogp

  def _make_general_head_with_mask(self, memory, mask, n_head):
    """
    :param memory: [None, num_units, encoding dim]
    :param n_head: output dim
    :param pd_type:
    :return:
    """
    # feature-wise max-pooling [Relational Deep RL, deepmind]
    head_h = tf.layers.max_pooling1d(memory,
                                     pool_size=self.tar_unit_dim,
                                     strides=1)
    head_h = tf.reshape(head_h, shape=[-1, self.enc_dim])
    head_h = tf.concat([head_h, mask, self.processed_x_vec], axis=-1)
    head_h = slim.fully_connected(head_h, 128, scope='fc1')
    head_h = slim.fully_connected(head_h, 128, scope='fc2')
    head_logits = slim.fully_connected(head_h,
                                       n_head,
                                       scope='logits',
                                       activation_fn=None,
                                       normalizer_fn=None)
    head_argmax = tf.argmax(head_logits, axis=-1)
    head_pd = self.head_pdtype.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    return head_logits, head_argmax, head_sam, head_neglogp

  def _make_multi_bi_head(self, memory, select_mask):
    """ memory: [None, unit_max_num, enc_dim] """
    x_vec_tmp = tf.expand_dims(self.processed_x_vec, axis=1)
    x_vec_tmp = tf.tile(x_vec_tmp, multiples=[1, tf.shape(memory)[1], 1])
    head_h = tf.concat([memory, x_vec_tmp], axis=-1)
    head_h = slim.fully_connected(head_h, 128, scope='fc1')
    head_h = slim.fully_connected(head_h, 128, scope='fc2')
    head_logits = slim.fully_connected(head_h,
                                       2,
                                       scope='logits',
                                       activation_fn=None,
                                       normalizer_fn=None)

    # modify the logits that unavailable position will be -inf
    neginf = tf.zeros_like(select_mask) - INF
    _mask = tf.equal(select_mask, 0)
    offset = tf.multiply(tf.cast(_mask, dtype=tf.float32), neginf)
    offset = tf.expand_dims(offset, axis=-1)
    offset = tf.concat([tf.zeros_like(offset), offset], axis=-1)
    head_logits += offset

    head_argmax = tf.argmax(head_logits, axis=-1)
    head_pd = self.head_pdtype.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    return head_logits, head_argmax, head_sam, head_neglogp

  def _make_ptr_head(self, query, ptr_mask, memory):
    query_h = slim.fully_connected(query, 128, scope='query_fc1')
    # concat query with the global vector feature
    query_h = tf.concat([query_h, self.processed_x_vec], axis=-1)
    query_h = slim.fully_connected(query_h, self.enc_dim, scope='query_fc2')
    query_h = tf.expand_dims(query_h, axis=1)

    select_logits, select_prob = self.decode(query_h, memory, training=True)
    select_logits = tf.reshape(select_logits, [-1, self.tar_unit_dim])

    # modify the logits that unavailable position will be -inf
    neginf = tf.zeros_like(select_logits) - INF
    _mask = tf.equal(ptr_mask, 0)
    select_logits = tf.where(_mask, neginf, select_logits)

    select_argmax = tf.argmax(select_logits, axis=-1)
    select_pd = self.head_pdtype.pdfromflat(select_logits)
    select_sam = select_pd.sample()
    select_neglogp = select_pd.neglogp(select_sam)

    return select_logits, select_argmax, select_sam, select_neglogp

  def encode(self, x, len_mask, training=True):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      enc = x
      # Blocks
      for i in range(self.num_enc_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # self-attention
          enc = multihead_attention(queries=enc,
                                    keys=enc,
                                    values=enc,
                                    dropout_rate=self.dropout_rate,
                                    training=training,
                                    causality=False)
          # feed forward
          enc = ff(enc, num_units=[self.ff_dim, self.enc_dim])
    memory = enc
    len_mask = tf.expand_dims(len_mask, axis=-1)
    memory = tf.multiply(len_mask, memory)
    return memory

  def decode(self, y, memory, training=True):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      dec_logits, dec_pd = [], []
      dec = y
      # Blocks
      for i in range(self.num_dec_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # Masked self-attention (Note that causality is True at this time)
          dec = multihead_attention(queries=dec,
                                    keys=dec,
                                    values=dec,
                                    dropout_rate=self.dropout_rate,
                                    training=training,
                                    causality=True,
                                    scope="self_attention")

          if i < self.num_dec_blocks - 1:
            # Vanilla attention
            dec = multihead_attention(queries=dec,
                                      keys=memory,
                                      values=memory,
                                      dropout_rate=self.dropout_rate,
                                      training=training,
                                      causality=False,
                                      scope="vanilla_attention")
            # Feed Forward
            dec = ff(dec, num_units=[self.ff_dim, self.enc_dim])
          else:
            # pointer attention
            dec_logits, dec_pd = multihead_attention(
              queries=dec,
              keys=memory,
              values=memory,
              dropout_rate=self.dropout_rate,
              training=training,
              causality=False,
              pointer=True,
              scope="pointer_attention")
    return dec_logits, dec_pd

  def _process_mask(self, mask):
    return tf.cast(mask, tf.float32)

  def _mask_logits(self, logits, mask):
    neginf = tf.zeros_like(logits) - INF
    _mask = tf.equal(mask, 0)
    logits = tf.where(_mask, neginf, logits)
    return logits


def transformer_test():
  nbatch = 32

  unit_num_max = 600
  feat_dim = 128
  img_size = (8, 8, 4)
  ob_list_space = spaces.Box(low=0, high=1, shape=(unit_num_max, feat_dim),
                             dtype=float)
  ob_vec_space = spaces.Box(low=0, high=1, shape=(10,), dtype=float)
  ob_img_space = spaces.Box(low=0, high=1, shape=img_size, dtype=float)
  ob_space = spaces.Tuple([ob_list_space, ob_vec_space])

  from timitate.lib.pb2action_converter import PB2ActionConverter
  ac_space = PB2ActionConverter().space
  sess = tf.Session()
  model = Transformer(sess, ob_space, ac_space)
  print('Successfully build the model.')

  tf.global_variables_initializer().run(session=sess)
  import joblib
  params = tf.trainable_variables()
  # for p in params:
  #   print(p)
  ps = sess.run(params)
  joblib.dump(ps, 'full_trans.ckpt')

  # state = np.zeros([1, unit_num_max, feat_dim])
  # masks_feed = {}
  # masks_feed['len'] = np.zeros([1, 600])
  # masks_feed['select'] = np.zeros([1, 137, 600])
  # masks_feed['target'] = np.zeros([1, 137, 600])
  # masks_feed['head'] = np.zeros([1, 137, 8])
  # masks_feed['ability'] = np.zeros([1, 137])
  #
  # ob = [state, masks_feed]
  # rl_outputs = model.step(ob)
  # a, v, initial_state, neglogp = rl_outputs
  # print('ability: {}'.format(a[0][0]))
  # print('shift: {}'.format(a[1][0]))
  # print('noop_num: {}'.format(a[2][0]))
  # print('tar_unit: {}'.format(a[3][0]))
  # print('tar_x: {}'.format(a[4][0]))
  # print('tar_y: {}'.format(a[5][0]))
  # print('s_select: {}'.format(a[6][0]))
  # print('m_select: {}'.format(a[7][0]))


if __name__ == '__main__':
  transformer_test()
