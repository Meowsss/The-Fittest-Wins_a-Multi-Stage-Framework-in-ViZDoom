from tpolicies.transformers.transformer_policy import *
from tpolicies.transformers.decent_team_transformer_policy import conv_encode, \
  team_transformer_test


class CentTeamTransformer(BasePolicy):
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
      a_argmax_tf, a_sam_tf, neglogp_tf, entropy_tf, pd_tf, vf_tf = \
        self._create_net(processed_team_x_list, processed_x_img)

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

  def _create_net(self, team_x, img_x):
    self.team_ac_pdtype = [CategoricalPdType(ncat=space.n)
                           for space in self.ac_space]
    team_logits = []
    team_argmax = []
    team_sam = []
    team_neglogp = []
    team_entropy = []
    team_pd = []

    # x representation
    team_memories = []
    memories = []
    for i, x in enumerate(team_x):
      x = slim.fully_connected(x, self.enc_dim, scope='x_fc1_team'+str(i))
      memory = self.trans_encode(x, training=True)  # [bs, num_agts, dim]
      memories.append(memory)
      # group set function: permutation invariant property in each group
      agg_memory = tf.reduce_sum(memory, axis=1)  # [bs, dim]
      team_memories.append(agg_memory)

    team_memories = tf.concat([tf.expand_dims(m, axis=1) for m in team_memories],
                              axis=1)  # [bs, num_teams, dim]
    team_memories = self.trans_encode(team_memories, training=True)
    team_memories = tf.unstack(team_memories, axis=1)

    vf = tf.concat(team_memories, axis=-1)  # [bs, dim * num_teams]
    vf = tf.concat([vf, img_x], axis=-1)  # [bs, dim * num_teams + img_vec_dim]
    vf = slim.fully_connected(vf, 128, scope='v_fc1')
    vf = slim.fully_connected(vf, 128, scope='v_fc2')
    vf = slim.fully_connected(vf, 1, scope='v_out', activation_fn=None,
                              normalizer_fn=None)

    for i, memory in enumerate(memories):
      team_memory = team_memories[i]
      team_memory = tf.expand_dims(team_memory, axis=1)
      team_embed = tf.concat([memory, tf.tile(team_memory,
                                              [1, tf.shape(memory)[1], 1])],
                             axis=-1)
      # concat embed per team with global img_x
      team_embed = tf.concat([team_embed, tf.tile(tf.expand_dims(img_x, axis=1),
                                                  [1, tf.shape(team_embed)[1], 1])],
                             axis=-1)

      team_embed = slim.fully_connected(team_embed, 128,
                                        scope='xx_fc1_team'+str(i))
      team_embed = slim.fully_connected(team_embed, 128,
                                        scope='xx_fc2_team'+str(i))
      team_embed = slim.fully_connected(team_embed,
                                        self.ac_space[i].n,
                                        scope='team_logits'+str(i),
                                        activation_fn=None,
                                        normalizer_fn=None)

      head_argmax = tf.argmax(team_embed, axis=-1)  # [batch_size, agent_num]
      head_pd = self.team_ac_pdtype[i].pdfromflat(team_embed)
      head_sam = head_pd.sample()
      head_neglogp = head_pd.neglogp(head_sam)
      head_entropy = head_pd.entropy()

      team_logits.append(team_embed)
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

    return team_argmax, team_sam, global_neglogp, global_entropy, team_pd, vf

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


if __name__ == '__main__':
  team_transformer_test(CentTeamTransformer, 'cent_trans.ckpt')
