from tpolicies.transformers.transformer_policy import *


class MarlTransformer(BasePolicy):
  def __init__(self, ob_space, ac_space, nbatch=None, reuse=False, env='5I',
               n_v=None, input_data=None, action_mask=False,
               scope_name="model"):
    # TODO: the arguments in the second line need to be checked; they follow the
    # requirement of ppo learner
    sess = tf.get_default_session()

    self.num_blocks = 3
    self.dropout_rate = 0.5
    self.per_agent_ac_dim = 9
    self.unit_max_len = 10

    self.env = env
    assert isinstance(ob_space, spaces.Tuple)
    self.ob_space = ob_space
    assert isinstance(ac_space, spaces.Box)
    self.ac_space = ac_space

    self.ff_dim = 64
    self.enc_dim = 128

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      if input_data is not None:
        X_list = input_data.X[0]
        processed_x_list = tf.to_float(X_list)
        len_mask = input_data.X[1]
        processed_len_mask = tf.to_float(len_mask)
      else:
        X_list, processed_x_list = gym_observation_input(
          ob_space=ob_space.spaces[0], input_data=input_data)
        len_mask, processed_len_mask = gym_observation_input(
          ob_space=ob_space.spaces[1], input_data=input_data)

      masks = {}
      masks['len'] = processed_len_mask
      self.masks = masks

      mha_argmax_tf, mha_tf, neglogp_tf, vf_tf, entropy_tf = \
        self._create_net(x=processed_x_list, masks=masks)

    self.initial_state = None

    def step(ob, *_args, **_kwargs):
      """ ob is a tuple of two: one is the state, one is the masks dict """
      state = ob[0][0]
      masks_feed = {}
      masks_feed['len'] = ob[0][1]

      state = np.expand_dims(state, axis=0)
      masks_feed['len'] = np.expand_dims(masks_feed['len'], axis=0)

      a, v, neglogp = sess.run([mha_tf, vf_tf, neglogp_tf],
                               {X_list: state,
                                masks['len']: masks_feed['len']
                                })
      return a, v, self.initial_state, neglogp

    def value(ob, *_args, **_kwargs):
      state = ob[0][0]
      masks_feed = {}
      masks_feed['len'] = ob[0][1]
      return sess.run(vf_tf,
                      {X_list: state,
                       masks['len']: masks_feed['len']
                       })

    # self.names = [X.name, head_a0.name, loc_a0.name, mov_seq_a0.name,
    # atk_seq_a0.name, vf.name, neglogp0.name]

    self.X_list = X_list
    self.vf = vf_tf
    self.a_argmax = mha_argmax_tf
    self.a = mha_tf
    self.neglogp = neglogp_tf
    self._entropy = entropy_tf
    self.step = step
    self.value = value
    PD = namedtuple('pd', ['neglogp', 'entropy'])
    self.pd = PD(self.neglogpac, self.entropy)

  def _create_net(self, x, masks):
    self.per_agt_ac_pdtype = CategoricalPdType(ncat=self.per_agent_ac_dim)

    # x representation
    with tf.variable_scope("enc"):
      x = slim.fully_connected(x, self.enc_dim, scope='x_fc1')
      memory = self.encode(x, masks['len'], training=True)

    with tf.variable_scope("head"):
      head_logits, head_argmax, head_sam, head_neglogp, head_entropy, head_pd = \
        self._make_head(memory=memory, select_mask=masks['len'],
                        pd_type=self.per_agt_ac_pdtype)
      self.head_pd = head_pd
      self.head_neglogp = head_neglogp
      self.head_logits = head_logits

    with tf.variable_scope("RL"):
      actions_sam = head_sam
      neglogp_final = tf.reduce_sum(head_neglogp, axis=-1)
      entropy_final = tf.reduce_sum(head_entropy, axis=-1)
      # create value
      memory_pooling = tf.layers.max_pooling1d(memory,
                                               pool_size=self.unit_max_len,
                                               strides=1)
      vf = tf.reshape(memory_pooling, shape=[-1, self.enc_dim])
      vf = slim.fully_connected(vf, 128, scope='v_fc1')
      vf = slim.fully_connected(vf, 128, scope='v_fc2')
      vf = slim.fully_connected(vf, 1, scope='v_out', activation_fn=None,
                                normalizer_fn=None)

    actions_argmax = head_argmax

    return actions_argmax, actions_sam, neglogp_final, vf, entropy_final

  def _make_head(self, memory, select_mask, pd_type):
    """ memory: [None, unit_max_num, enc_dim] """
    head_h = slim.fully_connected(memory, 128, scope='fc1')
    head_h = slim.fully_connected(head_h, 128, scope='fc2')
    head_logits = slim.fully_connected(head_h,
                                       self.per_agent_ac_dim,
                                       scope='logits',
                                       activation_fn=None,
                                       normalizer_fn=None)

    # modify the logits that unavailable position will be -inf
    neginf = tf.zeros_like(select_mask) - INF
    _mask = tf.equal(select_mask, 0)
    offset = tf.multiply(tf.cast(_mask, dtype=tf.float32), neginf)
    offset = tf.expand_dims(offset, axis=-1)
    offset = tf.concat([tf.zeros_like(offset)] +
                       [offset for _ in range(self.per_agent_ac_dim-1)],
                       axis=-1)
    head_logits += offset

    head_argmax = tf.argmax(head_logits, axis=-1)
    head_pd = pd_type.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    head_entropy = head_pd.entropy()
    return head_logits, head_argmax, head_sam, head_neglogp, head_entropy, head_pd

  def neglogpac(self, A):
    a_neglogpac = self.head_pd.neglogp(A)
    neglogpac_sum = tf.reduce_sum(a_neglogpac, axis=-1)

    return neglogpac_sum

  def entropy(self):
    return self._entropy

  def encode(self, x, len_mask, training=True):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      enc = x
      # Blocks
      for i in range(self.num_blocks):
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
    # len_mask = tf.expand_dims(len_mask, axis=-1)
    # memory = tf.multiply(len_mask, memory)
    return memory


def marl_transformer_test():
  unit_num_max = 10
  feat_dim = 6
  ob_space = spaces.Tuple(
    [spaces.Box(low=0, high=1, shape=(unit_num_max, feat_dim), dtype=float),
     spaces.Box(low=0, high=1, shape=(unit_num_max,), dtype=float)
     ])
  ac_space = spaces.Box(low=0, high=1, shape=(unit_num_max,), dtype=float)

  sess = tf.Session()
  with sess.as_default():
    model = MarlTransformer(ob_space, ac_space)

  print('Successfully build the model.')

  tf.global_variables_initializer().run(session=sess)

  state = np.zeros([unit_num_max, feat_dim])
  masks_feed = {}
  # masks_feed['len'] = np.ones([unit_num_max])
  masks_feed['len'] = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

  ob = [state, masks_feed['len']]
  a, v, initial_state, neglogp = model.step([ob])
  print('a: {}'.format(a[0]))


if __name__ == '__main__':
  marl_transformer_test()
