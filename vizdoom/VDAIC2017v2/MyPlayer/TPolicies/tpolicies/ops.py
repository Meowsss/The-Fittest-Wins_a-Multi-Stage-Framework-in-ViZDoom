""" operations extension. in style of tf.nn & tf ops"""
import tensorflow as tf


INF = 1e20


def mask_logits(logits, mask):
  neginf = tf.zeros_like(logits) - INF
  logits = tf.where(mask, logits, neginf)
  return logits


def mask_embed(embed, mask):
  mask = to_float32(tf.expand_dims(mask, axis=-1))
  return tf.multiply(embed, mask)


def to_float32(t):
  if t.dtype == tf.float32:
    return t
  return tf.cast(t, tf.float32)


def to_int32(t):
  if t.dtype == tf.int32:
    return t
  return tf.cast(t, tf.int32)


def to_bool(t):
  if t.dtype == tf.bool:
    return t
  return tf.cast(t, tf.bool)


def fetch_op(tensor, idx):
  """Fetch tensor given index

  Args:
    tensor: (bs, dim_a, dim_b, dim_c, ...)
    idx: (bs,),

  Returns:
    A tensor in shape (bs, dim_b, dim_c, ...)
  """
  return tf.gather_nd(tensor, tf.stack([tf.range(tf.shape(idx)[0]), idx], axis=1))


# rnn stuff
def batch_to_seq(inputs, nrollout, rollout_len, flat=False):
  """ Convert a Tensor to a Tensor Sequence (list of Tensors).

  Borrowed and modified from openai/baselines

  Args:
    inputs: (nrollout*rollout_len, d1, d2, ...)

  Returns:
    A list of Tensors, length rollout_len, each Tensor sized
    (nrollout, d1, d2, ...)
  """
  if flat:
    inputs = tf.reshape(inputs, [nrollout, rollout_len])
  else:
    inputs = tf.reshape(inputs, [nrollout, rollout_len, -1])
  return [tf.squeeze(v, [1]) for v in
          tf.split(axis=1, num_or_size_splits=rollout_len, value=inputs)]


def seq_to_batch(inputs: list, flat=False):
  """ Convert a Tensor Sequence (list of tensors) to a Tensor.

  Borrowed and modified from openai/baselines

  Args:
    inputs: a list, length rollout_len. Each Tensor sized
    (nrollout, d1, d2, ...)
    flat: boolean, whether flatten as vector

  Returns:
    A Tensor sized (nrollout*rollout_len, d1, d2, ...)
   """
  shape = inputs[0].get_shape().as_list()
  if not flat:
    assert len(shape) > 1, 'The rank ot the Tensor in inputs seq must be > 1'
    h_dims = inputs[0].get_shape().as_list()[1:]  # (d1, d2, ...)
    return tf.reshape(tf.concat(axis=1, values=inputs), [-1] + h_dims)
  else:
    return tf.reshape(tf.stack(values=inputs, axis=1), [-1])


def one_step_lstm_op(c, h, x, wx, wh, b, forget_bias, x_nf=None, h_nf=None,
                     c_nf=None):
  """ one step lstm op. """
  xx = tf.matmul(x, wx)
  xx = xx if x_nf is None else x_nf(xx)
  hh = tf.matmul(h, wh)
  hh = hh if h_nf is None else h_nf(hh)
  z = xx + hh + b
  i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
  i = tf.nn.sigmoid(i)
  f = tf.nn.sigmoid(f + forget_bias)
  o = tf.nn.sigmoid(o)
  u = tf.tanh(u)
  c = f * c + i * u
  cc = c if c_nf is None else c_nf(c)
  h = o * tf.tanh(cc)
  return c, h