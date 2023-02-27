def lstm(inputs_x_seq: list,
         inputs_mask_seq: list,
         inputs_state,
         nh,
         forget_bias=0.0,
         weights_initializer=ortho_init(1.0),
         biases_initializer=tf.constant_initializer(0.0),
         scope=None):
  """ lstm layer.
  TODO: doc that it is borrowed and modified from openai/baselines"""
  xs, ms, s = inputs_x_seq, inputs_mask_seq, inputs_state  # shorter names
  nbatch, nin = [v.value for v in xs[0].get_shape()]
  nsteps = len(xs)
  with tf.variable_scope(scope, default_name='lstm'):
    wx = tf.get_variable("wx", [nin, nh * 4], initializer=weights_initializer)
    wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer)
    b = tf.get_variable("b", [nh * 4], initializer=biases_initializer)

  c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
  for idx, (x, m) in enumerate(zip(xs, ms)):
    c = c * (1 - m)
    h = h * (1 - m)
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f + forget_bias)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    xs[idx] = h
  s = tf.concat(axis=1, values=[c, h])
  return xs, s


def lnlstm(inputs_x_seq: list,
           inputs_mask_seq: list,
           inputs_state,
           nh,
           forget_bias=0.0,
           weights_initializer=ortho_init(1.0),
           biases_initializer=tf.constant_initializer(0.0),
           scope=None):
  def _ln(_x, _g, _b, _e=1e-5, _axes=None):
    _axes = [1] if _axes is None else _axes
    u, s = tf.nn.moments(_x, axes=_axes, keep_dims=True)
    _x = (_x - u) / tf.sqrt(s + _e)
    _x = _x * _g + _b
    return _x

  nbatch, nin = [v.value for v in inputs_x_seq[0].get_shape()]
  with tf.variable_scope(scope, default_name='lnlstm'):

    wx = tf.get_variable("wx", [nin, nh * 4], initializer=weights_initializer)
    gammax = tf.get_variable("gx", [nh * 4],
                         initializer=tf.constant_initializer(1.0))
    betax = tf.get_variable("bx", [nh * 4], initializer=tf.zeros_initializer())

    wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer)
    gammah = tf.get_variable("gh", [nh * 4],
                         initializer=tf.constant_initializer(1.0))
    betah = tf.get_variable("bh", [nh * 4], initializer=tf.zeros_initializer())

    bias = tf.get_variable("b", [nh * 4], initializer=biases_initializer)

    gammac = tf.get_variable("gc", [nh],
                             initializer=tf.constant_initializer(1.0))
    betac = tf.get_variable("bc", [nh],
                            initializer=tf.zeros_initializer())

  c, h = tf.split(axis=1, num_or_size_splits=2, value=inputs_state)
  for idx, (x, m) in enumerate(zip(inputs_x_seq, inputs_mask_seq)):
    c = c * (1 - m)
    h = h * (1 - m)
    z = (_ln(tf.matmul(x, wx), gammax, betax) +
         _ln(tf.matmul(h, wh), gammah, betah) + bias)
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f + forget_bias)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(_ln(c, gammac, betac))
    inputs_x_seq[idx] = h
  inputs_state = tf.concat(axis=1, values=[c, h])
  return inputs_x_seq, inputs_state


def k_lstm(inputs_x_seq: list,
           inputs_mask_seq: list,
           inputs_state,
           nh,
           k=1,
           forget_bias=0.0,
           weights_initializer=ortho_init(1.0),
           biases_initializer=tf.constant_initializer(0.0),
           scope=None):
  """k-lstm layer."""

  def _make_cell(c, h, x, wx, wh, b):
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f + forget_bias)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    return c, h

  xs, ms, hs = inputs_x_seq, inputs_mask_seq, inputs_state  # shorter names

  # the last dim stores the step info
  step = hs[:, :, -1]  # [bs, rollout_len]
  # initial hidden state: hs0
  hs0 = hs[:, 0, :]  # [bs, 2*nlstm + 1]
  s, step0 = hs0[:, 0:-1], hs0[:, -1]  # [bs, 2*nlstm], [bs]
  nbatch, nin = [v.value for v in xs[0].get_shape()]
  rollout_len = len(xs)
  with tf.variable_scope(scope, default_name='k_lstm'):
    wx = tf.get_variable("wx", [nin, nh * 4], initializer=weights_initializer)
    wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer)
    b = tf.get_variable("b", [nh * 4], initializer=biases_initializer)

  c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
  for idx, (x, m) in enumerate(zip(xs, ms)):
    c = c * (1 - m)
    h = h * (1 - m)
    c_lstm, h_lstm = _make_cell(c, h, x, wx, wh, b)
    mod_mask = tf.equal(tf.mod(step[:, idx], k), 0)
    mod_mask = tf.expand_dims(mod_mask, axis=-1)
    mod_mask = tf.cast(mod_mask, tf.float32)
    c = tf.multiply(mod_mask, c_lstm) + tf.multiply(1 - mod_mask, c)
    h = tf.multiply(mod_mask, h_lstm) + tf.multiply(1 - mod_mask, h)
    xs[idx] = h
  s = tf.concat(axis=1, values=[c, h])
  # append the increased step
  s = tf.concat([s, tf.expand_dims(tf.mod(step0 + rollout_len, k), axis=-1)],
                axis=-1)
  return xs, s