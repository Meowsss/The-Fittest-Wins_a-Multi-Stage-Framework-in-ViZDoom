from tpolicies.transformers.transformer_policy import *


def conv_encode(x):
  # x = tf.transpose(x, perm=[0, 2, 3, 1])  # NCHW -> NHWC
  # encode
  x = slim.conv2d(x, 64, [3, 3], scope='conv1')
  x = slim.conv2d(x, 64, [3, 3], scope='conv2')
  x = slim.conv2d(x, 64, [3, 3], scope='conv3')
  # fc-out
  x = slim.max_pool2d(x, [2, 2], scope='pool1')
  x = slim.conv2d(x, 128, [3, 3], scope='conv_vf1')
  x = slim.max_pool2d(x, [2, 2], scope='pool2')
  x = slim.flatten(x)
  x = slim.fully_connected(x, 128, scope='fc1')
  return x


class DecentTeamTransformer(BasePolicy):
  def __init__(self, ob_space, ac_space, nbatch=None, reuse=False, env='5I',
               n_v=None, input_data=None, action_mask=False,
               scope_name="model"):
    # TODO: the arguments in the second line need to be checked; they follow the
    # requirement of ppo learner
    sess = tf.get_default_session()

    assert isinstance(ob_space, spaces.Tuple)
    self.ob_space = ob_space
    assert isinstance(ac_space, spaces.Tuple)
    self.ac_space = ac_space

    self.num_teams = len(ac_space.spaces)
    self.num_blocks = 3
    self.dropout_rate = 0.5

    self.ff_dim = 64
    self.enc_dim = 128

    with tf.variable_scope("model", reuse=reuse):
      team_x_list = []
      processed_team_x_list = []
      if input_data is not None:
        for i in range(self.num_teams):
          team_x_list.append(input_data.X[i])
          processed_team_x_list.append(tf.to_float(team_x_list))

        x_img = input_data.X[-1]
        processed_x_img = tf.to_float(input_data.X[-1])
      else:
        for i in range(self.num_teams):
          a, b = gym_observation_input(
            ob_space=ob_space.spaces[i], input_data=input_data)
          team_x_list.append(a)
          processed_team_x_list.append(b)

        x_img, processed_x_img = gym_observation_input(
          ob_space=ob_space.spaces[-1], input_data=input_data)

      processed_x_img = conv_encode(processed_x_img)  # [bs, dim]
      a_argmax_tf, a_sam_tf, neglogp_tf, entropy_tf, pd_tf = self._create_pi(
        processed_team_x_list, processed_x_img)
      vf_tf = self._create_value(processed_team_x_list, processed_x_img)

    self.initial_state = None

    def step(ob_list):
      feed_dict = {}
      for i, state in enumerate(ob_list):
        if i != len(ob_list) - 1:
          feed_dict[team_x_list[i]] = np.expand_dims(ob_list[i], axis=0)
      feed_dict[x_img] = np.expand_dims(ob_list[-1], axis=0)

      a, v, neglogp = sess.run([a_sam_tf, vf_tf, neglogp_tf],
                               feed_dict)
      return a, v, self.initial_state, neglogp

    def value(ob_list):
      feed_dict = {}
      for i, state in enumerate(ob_list):
        if i != len(ob_list) - 1:
          feed_dict[team_x_list[i]] = np.expand_dims(ob_list[i], axis=0)
      feed_dict[x_img] = np.expand_dims(ob_list[-1], axis=0)

      return sess.run(vf_tf, feed_dict)

    self.X_list = team_x_list
    self.vf = vf_tf
    self.a_argmax = a_argmax_tf
    self.a = a_sam_tf
    self.neglogp = neglogp_tf
    self._entropy = entropy_tf
    self.step = step
    self.value = value
    PD = namedtuple('pd', ['neglogp', 'entropy'])
    self.pd_list = pd_tf
    self.pd = PD(self.neglogpac, self.entropy)

  def _create_pi(self, team_x, img_x):
    self.team_ac_pdtype = [CategoricalPdType(ncat=space.n)
                           for space in self.ac_space]
    team_logits = []
    team_argmax = []
    team_sam = []
    team_neglogp = []
    team_entropy = []
    team_pd = []

    with tf.variable_scope("pi"):
      for i, x in enumerate(team_x):  # x = [batch_size, agent_num, dim]
        # concat x per team with global img_x
        x = tf.concat([x, tf.tile(tf.expand_dims(img_x, axis=1),
                                  [1, tf.shape(x)[1], 1])],
                      axis=-1)

        head_h = slim.fully_connected(x, self.enc_dim, scope='x_fc1_team'+str(i))
        head_h = slim.fully_connected(head_h, self.enc_dim, scope='x_fc2_team'+str(i))
        head_logits = slim.fully_connected(head_h,
                                           self.ac_space[i].n,
                                           scope='logits_team'+str(i),
                                           activation_fn=None,
                                           normalizer_fn=None)

        head_argmax = tf.argmax(head_logits, axis=-1)  # [batch_size, agent_num]
        head_pd = self.team_ac_pdtype[i].pdfromflat(head_logits)
        head_sam = head_pd.sample()
        head_neglogp = head_pd.neglogp(head_sam)
        head_entropy = head_pd.entropy()

        team_logits.append(head_logits)
        team_argmax.append(head_argmax)
        team_pd.append(head_pd)
        team_sam.append(head_sam)
        team_neglogp.append(head_neglogp)
        team_entropy.append(head_entropy)

      global_neglogp = [tf.reduce_sum(per_team_neglogp, axis=-1)
                        for per_team_neglogp in team_neglogp]
      global_neglogp = tf.add_n(global_neglogp)  # [batch_size,]

      global_entropy = [tf.reduce_sum(per_team_entropy, axis=-1)
                        for per_team_entropy in team_entropy]
      global_entropy = tf.add_n(global_entropy)

    return team_argmax, team_sam, global_neglogp, global_entropy, team_pd

  def _create_value(self, team_x, img_x):
    team_memory = []
    # x representation
    with tf.variable_scope("v_enc"):
      for i, x in enumerate(team_x):
        x = slim.fully_connected(x, self.enc_dim, scope='x_fc1_team'+str(i))
        memory = self.trans_encode(x, training=True)  # [bs, num_agts, dim]
        # group set function: permutation invariant property in each group
        memory = tf.reduce_sum(memory, axis=1)  # [bs, dim]
        team_memory.append(memory)

    vf = tf.concat(team_memory, axis=-1)  # [bs, dim * num_teams]
    vf = tf.concat([vf, img_x], axis=-1)  # [bs, dim * num_teams + img_vec_dim]
    vf = slim.fully_connected(vf, 128, scope='v_fc1')
    vf = slim.fully_connected(vf, 128, scope='v_fc2')
    vf = slim.fully_connected(vf, 1, scope='v_out', activation_fn=None,
                              normalizer_fn=None)

    return vf

  def neglogpac(self, team_A):
    neglogpac_sum = []
    for i, a in enumerate(team_A):
      per_team_neglogpac = self.pd_list[i].neglogp(team_A[i])
      neglogpac_sum.append(tf.reduce_sum(per_team_neglogpac, axis=-1))

    neglogpac_sum = tf.add_n(neglogpac_sum)
    return neglogpac_sum

  def entropy(self):
    return self._entropy

  def trans_encode(self, x, training=True):
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


def team_transformer_test(Model, name='decent_trans.ckpt'):
  num_teams = 2
  max_unit_per_team = 10
  feat_dim = 6

  ob_space = spaces.Tuple(
    [spaces.Box(low=0, high=1, shape=(max_unit_per_team, feat_dim), dtype=float),
     spaces.Box(low=0, high=1, shape=(max_unit_per_team, feat_dim), dtype=float),
     spaces.Box(low=0, high=1, shape=(8, 8, num_teams * 2), dtype=float)
     ])
  ac_space = spaces.Tuple(
    [spaces.Discrete(9),
     spaces.Discrete(9)
     ])

  sess = tf.Session()
  with sess.as_default():
    model = Model(ob_space, ac_space)

  print('Successfully build the model.')

  tf.global_variables_initializer().run(session=sess)
  import joblib
  params = tf.trainable_variables()
  ps = sess.run(params)
  joblib.dump(ps, name)

  ob = [np.zeros([max_unit_per_team, feat_dim]) for _ in range(num_teams)] + \
       [np.zeros([8, 8, num_teams*2])]

  a, v, initial_state, neglogp = model.step(ob)
  print('a: {}'.format(a))
  print('neglogp: {}'.format(neglogp))
  print('v: {}'.format(v))


if __name__ == '__main__':
  team_transformer_test(DecentTeamTransformer)
