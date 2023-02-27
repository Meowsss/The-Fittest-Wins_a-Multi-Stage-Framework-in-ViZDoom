from tpolicies.transformers.transformer_policy import *


class MiniTransformer(BasePolicy):
  def __init__(self, ob_space, ac_space, nbatch=None, reuse=False, env='5I',
               n_v=None, input_data=None, action_mask=False,
               scope_name="model"):
    # TODO: the arguments in the second line need to be checked; they follow the
    # requirement of ppo learner
    sess = tf.get_default_session()

    self.num_blocks = 3
    self.dropout_rate = 0.5
    self.num_act_heads = 8

    self.env = env
    assert isinstance(ob_space, spaces.Tuple)
    self.ob_space = ob_space
    assert isinstance(ac_space, spaces.Tuple)
    self.ac_space = ac_space

    for i in range(len(ac_space.spaces)):
      if i < len(ac_space.spaces) - 2:
        assert isinstance(ac_space.spaces[i], spaces.Discrete)
      else:
        assert isinstance(ac_space.spaces[i], spaces.Box)

    self.ability_dim = ac_space.spaces[0].n
    self.tar_unit_dim = ac_space.spaces[1].n
    self.tar_loc_x_dim = ac_space.spaces[2].n
    self.tar_loc_y_dim = ac_space.spaces[3].n
    self.select_dim = ac_space.spaces[4].shape[0]
    assert ac_space.spaces[5].shape[0] == ac_space.spaces[4].shape[0]

    self.ff_dim = 64
    self.enc_dim = 128

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      # X_list, processed_x_list = observation_input(ob_space.spaces[0], nbatch)
      # X_img, processed_x_img = observation_input(ob_space.spaces[1], nbatch)
      if input_data is not None:
        X_list = input_data.X[0]
        processed_x_list = tf.to_float(X_list)
        len_mask = input_data.X[1]
        processed_len_mask = tf.to_float(len_mask)
        enemy_mask = input_data.X[2]
        processed_enemy_mask = tf.to_float(enemy_mask)
        self_mask = input_data.X[3]
        processed_self_mask = tf.to_float(self_mask)
      else:
        X_list, processed_x_list = gym_observation_input(
          ob_space=ob_space.spaces[0], input_data=input_data)
        len_mask, processed_len_mask = gym_observation_input(
          ob_space=ob_space.spaces[1], input_data=input_data)
        enemy_mask, processed_enemy_mask = gym_observation_input(
          ob_space=ob_space.spaces[2], input_data=input_data)
        self_mask, processed_self_mask = gym_observation_input(
          ob_space=ob_space.spaces[3], input_data=input_data)

      masks = {}
      masks['len'] = processed_len_mask
      masks['select'] = processed_self_mask
      masks['target'] = processed_enemy_mask
      self.masks = masks

      mha_argmax_tf, mha_tf, neglogp_tf, vf_tf, entropy_tf = \
        self._create_net(x=processed_x_list, masks=masks)
      # ability0, shift0, tar_unit0, \
      # tar_loc_x0, tar_loc_y0, s_select0, m_select0 = a0
      # vf = fc(vf_h, 'vf', 1)[:, 0]

    self.initial_state = None

    def step(ob, *_args, **_kwargs):
      """ ob is a tuple of two: one is the state, one is the masks dict """
      state = ob[0][0]
      masks_feed = {}
      masks_feed['len'] = ob[0][1]
      masks_feed['target'] = ob[0][2]
      masks_feed['select'] = ob[0][3]

      state = np.expand_dims(state, axis=0)
      masks_feed['len'] = np.expand_dims(masks_feed['len'], axis=0)
      masks_feed['target'] = np.expand_dims(masks_feed['target'], axis=0)
      masks_feed['select'] = np.expand_dims(masks_feed['select'], axis=0)

      assert len(np.shape(state)) == 3

      a, v, neglogp = sess.run([mha_tf, vf_tf, neglogp_tf],
                               {X_list: state,
                                masks['len']: masks_feed['len'],
                                masks['select']: masks_feed['select'],
                                masks['target']: masks_feed['target']
                                })
      # print('v: {}, neglogp: {}'.format(v, neglogp))
      # print('feat_loc_x: {}, feat_loc_y: {}'.format(a[2], a[3]))
      return a, v, self.initial_state, neglogp

    def value(ob, *_args, **_kwargs):
      state = ob[0][0]
      masks_feed = {}
      masks_feed['len'] = ob[0][1]
      masks_feed['target'] = ob[0][2]
      masks_feed['select'] = ob[0][3]
      return sess.run(vf_tf,
                      {X_list: state,
                       masks['len']: masks_feed['len'],
                       masks['select']: masks_feed['select'],
                       masks['target']: masks_feed['target']
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
    self.ability_pdtype = CategoricalPdType(ncat=self.ability_dim)
    self.loc_x_pdtype = CategoricalPdType(ncat=self.tar_loc_x_dim)
    self.loc_y_pdtype = CategoricalPdType(ncat=self.tar_loc_y_dim)
    self.ptr_pdtype = CategoricalPdType(ncat=self.tar_unit_dim)
    self.bi_pdtype = CategoricalPdType(ncat=2)

    # x representation
    with tf.variable_scope("enc"):
      x = slim.fully_connected(x, self.enc_dim, scope='x_fc1')
      memory = self.encode(x, masks['len'], training=True)

    """ begin multi-head """
    # ability
    with tf.variable_scope("ability"):
      ability_logits, ability_argmax, ability_sam, \
      ability_neglogp, ability_entropy, ability_pd = \
        self._make_general_head(memory, self.ability_dim,
                                pd_type=self.ability_pdtype)
      self.ability_pd = ability_pd
      self.ability_neglogp = ability_neglogp
    """ below requires the auto-regressive input """
    # tar_unit
    with tf.variable_scope("tar_unit"):
      tar_unit_logits, tar_unit_argmax, tar_unit_sam, \
      tar_unit_neglogp, tar_unit_entropy, tar_unit_pd = \
        self._make_ptr_head(memory=memory,
                            ptr_mask=masks['target'],
                            pd_type=self.ptr_pdtype)
      self.tar_unit_pd = tar_unit_pd
      self.tar_unit_neglogp = tar_unit_neglogp
      self.tar_unit_logits = tar_unit_logits
    # tar_loc_x
    with tf.variable_scope("tar_loc_x"):
      tar_loc_x_logits, tar_loc_x_argmax, tar_loc_x_sam, \
      tar_loc_x_neglogp, tar_loc_x_entropy, tar_loc_x_pd = \
        self._make_general_head(memory, self.tar_loc_x_dim,
                                pd_type=self.loc_x_pdtype)
      self.tar_loc_x_pd = tar_loc_x_pd
      self.tar_loc_x_neglogp = tar_loc_x_neglogp
    # tar_loc_y
    with tf.variable_scope("tar_loc_y"):
      tar_loc_y_logits, tar_loc_y_argmax, tar_loc_y_sam, \
      tar_loc_y_neglogp, tar_loc_y_entropy, tar_loc_y_pd = \
        self._make_general_head(memory, self.tar_loc_y_dim,
                                pd_type=self.loc_y_pdtype)
      self.tar_loc_y_pd = tar_loc_y_pd
      self.tar_loc_y_neglogp = tar_loc_y_neglogp
    # multi_select
    with tf.variable_scope("select_mov"):
      m_select_logits, m_select_argmax, m_select_sam, \
      m_select_neglogp, m_select_entropy, m_select_pd = \
        self._make_multi_bi_head(memory=memory, select_mask=masks['select'],
                                 pd_type=self.bi_pdtype)
      self.m_select_pd = m_select_pd
      self.m_select_neglogp = m_select_neglogp
      self.m_select_logits = m_select_logits
    with tf.variable_scope("select_atk"):
      a_select_logits, a_select_argmax, a_select_sam, \
      a_select_neglogp, a_select_entropy, a_select_pd = \
        self._make_multi_bi_head(memory=memory, select_mask=masks['select'],
                                 pd_type=self.bi_pdtype)
      self.a_select_pd = a_select_pd
      self.a_select_neglogp = a_select_neglogp
      self.a_select_logits = a_select_logits

    with tf.variable_scope("RL"):
      # create final sampled actions
      actions_sam = [ability_sam, tar_unit_sam,
                     tar_loc_x_sam, tar_loc_y_sam,
                     m_select_sam, a_select_sam]
      # create final neglogp
      """ neglogp will be used to compute a ratio,
      so the (masked) constant cares """
      tar_unit_flat_neglogp = tf.zeros_like(tar_unit_neglogp) - \
                              np.log(1.0 / self.tar_unit_dim)
      tar_x_flat_neglogp = tf.zeros_like(tar_loc_x_neglogp) - \
                           np.log(1.0 / self.tar_loc_x_dim)
      tar_y_flat_neglogp = tf.zeros_like(tar_loc_y_neglogp) - \
                           np.log(1.0 / self.tar_loc_y_dim)

      ability_sam = tf.stop_gradient(ability_sam)
      ability_sam_bool = tf.equal(ability_sam, 1)
      ability_sam = tf.cast(ability_sam, tf.float32)
      neglogp_final = ability_neglogp + \
                      tf.where(ability_sam_bool, tar_unit_neglogp,
                               tar_unit_flat_neglogp) + \
                      tf.where(ability_sam_bool, tar_x_flat_neglogp,
                               tar_loc_x_neglogp) + \
                      tf.where(ability_sam_bool, tar_y_flat_neglogp,
                               tar_loc_y_neglogp) + \
                      tf.multiply(1 - ability_sam,
                                  tf.reduce_sum(m_select_neglogp, axis=-1)) + \
                      tf.multiply(ability_sam,
                                  tf.reduce_sum(a_select_neglogp, axis=-1))

      """ entropy is directly added in the loss,
      so the constant can be arbitrary """
      entropy_final = ability_entropy + \
                      tf.multiply(ability_sam, tar_unit_entropy) + \
                      tf.multiply(1 - ability_sam, tar_loc_x_entropy) + \
                      tf.multiply(1 - ability_sam, tar_loc_y_entropy) + \
                      tf.multiply(1 - ability_sam,
                                  tf.reduce_sum(m_select_entropy, axis=-1)) + \
                      tf.multiply(ability_sam,
                                  tf.reduce_sum(a_select_entropy, axis=-1))
      # create value
      memory_pooling = tf.layers.max_pooling1d(memory,
                                               pool_size=self.tar_unit_dim,
                                               strides=1)
      vf = tf.reshape(memory_pooling, shape=[-1, self.enc_dim])
      vf = slim.fully_connected(vf, 128, scope='v_fc1')
      vf = slim.fully_connected(vf, 128, scope='v_fc2')
      vf = slim.fully_connected(vf, 1, scope='v_out', activation_fn=None,
                                normalizer_fn=None)

    actions_argmax = [ability_argmax, tar_unit_argmax,
                      tar_loc_x_argmax, tar_loc_y_argmax,
                      m_select_argmax, a_select_argmax]

    return actions_argmax, actions_sam, neglogp_final, vf, entropy_final

  def _make_general_head(self, memory, n_head, pd_type):
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
    head_h = slim.fully_connected(head_h, 128, scope='fc1')
    head_h = slim.fully_connected(head_h, 128, scope='fc2')
    head_logits = slim.fully_connected(head_h,
                                       n_head,
                                       scope='logits',
                                       activation_fn=None,
                                       normalizer_fn=None)
    head_argmax = tf.argmax(head_logits, axis=-1)
    head_pd = pd_type.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    head_entropy = head_pd.entropy()
    return head_logits, head_argmax, head_sam, head_neglogp, head_entropy, head_pd

  def _make_multi_bi_head(self, memory, select_mask, pd_type):
    """ memory: [None, unit_max_num, enc_dim] """
    head_h = slim.fully_connected(memory, 128, scope='fc1')
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
    head_pd = pd_type.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    head_entropy = head_pd.entropy()
    return head_logits, head_argmax, head_sam, head_neglogp, head_entropy, head_pd

  def _make_ptr_head(self, memory, ptr_mask, pd_type):
    """
    :param memory: [None, num_units, encoding dim]
    :param n_head: output dim
    :param pd_type:
    :return:
    """
    head_h = slim.fully_connected(memory, 128, scope='query_fc1')
    head_h = slim.fully_connected(head_h, 128, scope='query_fc2')
    select_logits = slim.fully_connected(head_h,
                                         1,
                                         scope='logits',
                                         activation_fn=None,
                                         normalizer_fn=None)

    select_logits = tf.reshape(select_logits, shape=[-1, self.tar_unit_dim])

    # modify the logits that unavailable position will be -inf
    neginf = tf.zeros_like(select_logits) - INF
    _mask = tf.equal(ptr_mask, 0)
    select_logits = tf.where(_mask, neginf, select_logits)

    select_argmax = tf.argmax(select_logits, axis=-1)
    select_pd = pd_type.pdfromflat(select_logits)
    select_sam = select_pd.sample()
    select_neglogp = select_pd.neglogp(select_sam)
    select_entropy = select_pd.entropy()
    return select_logits, select_argmax, select_sam, select_neglogp, select_entropy, select_pd

  def neglogpac(self, A):
    ability_sam = A[0]
    tar_unit_sam = A[1]
    tar_loc_x_sam = A[2]
    tar_loc_y_sam = A[3]
    m_select_sam = A[4]
    a_select_sam = A[5]

    ability_neglogpac = self.ability_pd.neglogp(ability_sam)
    tar_unit_neglogpac = self.tar_unit_pd.neglogp(tar_unit_sam)
    tar_loc_x_neglogpac = self.tar_loc_x_pd.neglogp(tar_loc_x_sam)
    tar_loc_y_neglogpac = self.tar_loc_y_pd.neglogp(tar_loc_y_sam)
    m_select_neglogpac = self.m_select_pd.neglogp(m_select_sam)
    a_select_neglogpac = self.a_select_pd.neglogp(a_select_sam)

    tar_unit_flat_neglogpac = tf.zeros_like(tar_unit_neglogpac) - \
                              np.log(1.0 / self.tar_unit_dim)
    tar_x_flat_neglogpac = tf.zeros_like(tar_loc_x_neglogpac) - \
                           np.log(1.0 / self.tar_loc_x_dim)
    tar_y_flat_neglogpac = tf.zeros_like(tar_loc_y_neglogpac) - \
                           np.log(1.0 / self.tar_loc_y_dim)

    ability_sam = tf.stop_gradient(ability_sam)
    ability_sam_bool = tf.equal(ability_sam, 1)
    ability_sam = tf.cast(ability_sam, tf.float32)
    neglogpac_sum = ability_neglogpac + \
                    tf.where(ability_sam_bool, tar_unit_neglogpac,
                             tar_unit_flat_neglogpac) + \
                    tf.where(ability_sam_bool, tar_x_flat_neglogpac,
                             tar_loc_x_neglogpac) + \
                    tf.where(ability_sam_bool, tar_y_flat_neglogpac,
                             tar_loc_y_neglogpac) + \
                    tf.multiply(1.0 - ability_sam,
                                tf.reduce_sum(m_select_neglogpac, axis=-1)) + \
                    tf.multiply(ability_sam,
                                tf.reduce_sum(a_select_neglogpac, axis=-1))
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
    len_mask = tf.expand_dims(len_mask, axis=-1)
    memory = tf.multiply(len_mask, memory)
    return memory


def test_mini_transformer():
  nbatch = 32

  unit_num_max = 10
  feat_dim = 6
  ob_space = spaces.Tuple(
    [spaces.Box(low=0, high=1, shape=(unit_num_max, feat_dim), dtype=float),
     spaces.Box(low=0, high=1, shape=(unit_num_max,), dtype=float),
     spaces.Box(low=0, high=1, shape=(unit_num_max,), dtype=float),
     spaces.Box(low=0, high=1, shape=(unit_num_max,), dtype=float)
     ])
  ac_space = spaces.Tuple([spaces.Discrete(2),
                           spaces.Discrete(10),
                           spaces.Discrete(8),
                           spaces.Discrete(8),
                           spaces.Box(low=0, high=1, shape=(unit_num_max,),
                                      dtype=float),
                           spaces.Box(low=0, high=1, shape=(unit_num_max,),
                                      dtype=float)
                           ])

  sess = tf.Session()
  with sess.as_default():
    model = MiniTransformer(ob_space, ac_space)

  print('Successfully build the model.')

  tf.global_variables_initializer().run(session=sess)

  state = np.zeros([unit_num_max, feat_dim])
  masks_feed = {}
  masks_feed['len'] = np.ones([unit_num_max])
  masks_feed['select'] = np.zeros([unit_num_max])
  masks_feed['target'] = np.ones([unit_num_max])

  ob = [state, masks_feed['len'], masks_feed['target'], masks_feed['select']]
  a, v, initial_state, neglogp = model.step([ob])
  print('ability: {}'.format(a[0][0]))
  print('tar_unit: {}'.format(a[1][0]))
  print('tar_x: {}'.format(a[2][0]))
  print('tar_y: {}'.format(a[3][0]))
  print('m_select: {}'.format(a[4][0]))
  print('a_select: {}'.format(a[5][0]))


if __name__ == '__main__':
  test_mini_transformer()
