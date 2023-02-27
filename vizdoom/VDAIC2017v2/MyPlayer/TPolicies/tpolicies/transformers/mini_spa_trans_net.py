from tpolicies.transformers.transformer_policy import *
from timitate.meta.utils import load_ability_info
a = load_ability_info()


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


Head = namedtuple('HEAD', ['logits', 'argmax', 'sam', 'neglogp', 'pd', 'ent'])
Heads = namedtuple('HEADS', ['b_ability', 'shift', 'noop', 'tar_unit',
                             'tar_loc', 's_select', 'm_select', 'camera'])


class SpatialTrans(BasePolicy):
  def __init__(self, ob_space=None, ac_space=None, sess=None,
               nbatch=None, reuse=tf.AUTO_REUSE,
               input_data=None, test=False,
               n_v=1, action_mask=None, rl=True, scope_name='model'):
    if sess is None:
      sess = tf.get_default_session()

    self.prints = []
    self.test = test
    self.rl = rl
    self.num_enc_blocks = 1
    self.num_dec_blocks = 1
    self.dropout_rate = 0.5
    self.num_act_heads = 7

    assert isinstance(ob_space, spaces.Tuple)
    # assert len(ob_space) == 3
    self.ob_space = ob_space
    # assert len(self.ob_space[-1].shape) == 3
    assert isinstance(ac_space, spaces.Tuple)
    self.ac_space = ac_space

    self.map_r_max = ob_space.spaces[2].shape[0]  # 176
    self.map_c_max = ob_space.spaces[2].shape[1]  # 200
    print('obs_space: {}'.format(ob_space))
    print('act_space: {}'.format(ac_space))

    for i in range(len(ac_space.spaces)):
      if i != 7:
        assert isinstance(ac_space.spaces[i], spaces.Discrete)
      else:
        assert isinstance(ac_space.spaces[i], spaces.Box)

    self.ability_dim = ac_space.spaces[0].n
    self.shift_dim = ac_space.spaces[1].n
    self.noop_dim = ac_space.spaces[2].n
    self.tar_unit_dim = ac_space.spaces[3].n
    if len(ac_space.spaces) == 8:
      self.tar_loc_dim = ac_space.spaces[4].n * ac_space.spaces[5].n
    else:
      self.tar_loc_dim = ac_space.spaces[4].n
    self.select_dim = self.tar_unit_dim

    self.ff_dim = 32
    self.enc_dim = 32
    self.max_bases_num = 16

    masks, labels = {}, {}
    self.input_data = input_data
    if input_data is None:
      X_list, processed_x_list = my_observation_input(ob_space.spaces[0],
                                                      nbatch)
      X_vec, processed_x_vec = my_observation_input(ob_space.spaces[1], nbatch)
      X_img, processed_x_img = my_observation_input(ob_space.spaces[2], nbatch)

      masks['len'] = tf.placeholder(shape=(nbatch, self.select_dim),
                                    dtype=tf.float32, name='len_mask')
      masks['ability'] = tf.placeholder(shape=(nbatch, self.ability_dim),
                                        dtype=tf.float32, name='ability_mask')
      masks['head'] = tf.placeholder(
        shape=(nbatch, self.ability_dim, self.num_act_heads),
        dtype=tf.float32, name='head_mask')
      masks['select'] = tf.placeholder(
        shape=(nbatch, self.ability_dim, self.select_dim),
        dtype=tf.float32, name='select_mask')
      masks['target'] = tf.placeholder(
        shape=(nbatch, self.ability_dim, self.select_dim),
        dtype=tf.float32, name='target_mask')
      masks['build'] = tf.placeholder(
        shape=(nbatch, self.map_r_max, self.map_c_max),
        dtype=tf.float32, name='build_mask')
      masks['base_pos'] = tf.placeholder(
        shape=(nbatch, self.max_bases_num, 2),
        dtype=tf.float32, name='base_pos_mask')
      labels['ability'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                         name='ability_labels')
      labels['shift'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                       name='shift_labels')
      labels['noop_num'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='noop_num_labels')
      labels['tar_unit'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='tar_unit_labels')
      labels['tar_loc_x'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                           name='tar_loc_x_labels')
      labels['tar_loc_y'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                           name='tar_loc_y_labels')
      labels['s_select'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='s_select_labels')
      labels['m_select'] = tf.placeholder(shape=(nbatch, self.select_dim),
                                          dtype=tf.int32,
                                          name='m_select_labels')
      labels['camera_x'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='camera_x_labels')
      labels['camera_y'] = tf.placeholder(shape=(nbatch,), dtype=tf.int32,
                                          name='camera_y_labels')
    else:
      if rl:
        processed_x_list = input_data.X[0]
        X_list = processed_x_list
        processed_x_vec = input_data.X[1]
        X_vec = processed_x_vec
        processed_x_img = input_data.X[2]
        X_img = processed_x_img

        masks['ability'] = self._process_to_float32(input_data.X[3])
        masks['head'] = self._process_to_float32(input_data.X[4])
        masks['select'] = self._process_to_float32(input_data.X[5])
        masks['target'] = self._process_to_float32(input_data.X[6])
        masks['len'] = self._process_to_float32(input_data.X[7])
        masks['build'] = self._process_to_float32(input_data.X[8])
        masks['base_pos'] = self._process_to_float32(input_data.X[9])
        self.masks = masks

        labels['ability'], labels['shift'], labels['noop_num'], labels['tar_unit'],\
        A_loc, labels['s_select'], labels['m_select'], A_camera = input_data.A
        # (x,y) in imitation and (x * self.map_c_max + y) in RL
        labels['tar_loc_x'], labels['tar_loc_y'] = self._loc_to_xy(A_loc)
        labels['camera_x'], labels['camera_y'] = self._loc_to_xy(A_camera)
      else:
        processed_x_list = self._process_to_float32(input_data.X_UNITS)
        X_list = processed_x_list
        processed_x_vec = self._process_to_float32(input_data.X_GLOBAL)
        X_vec = processed_x_vec
        processed_x_img = self._process_to_float32(input_data.X_IMAGE)
        X_img = processed_x_img

        masks['ability'] = self._process_to_float32(input_data.ABILITY_MASKS)
        masks['head'] = self._process_to_float32(input_data.HEAD_MASKS)
        masks['select'] = self._process_to_float32(input_data.SELECT_MASKS)
        masks['target'] = self._process_to_float32(input_data.TARGET_MASKS)
        masks['len'] = self._process_to_float32(input_data.LEN_MASKS)
        masks['build'] = self._process_to_float32(input_data.BUILD_MASKS)
        masks['base_pos'] = self._process_to_float32(input_data.BASE_POS_MASKS)

        labels['ability'] = input_data.ABILITY_LABELS
        labels['shift'] = input_data.SHIFT_LABELS
        labels['noop_num'] = input_data.NOOP_NUM_LABELS
        labels['tar_unit'] = input_data.TAR_UNIT_LABELS
        labels['tar_loc_x'] = input_data.TAR_X_LABELS
        labels['tar_loc_y'] = input_data.TAR_Y_LABELS
        labels['s_select'] = input_data.S_SELECT_LABELS
        labels['m_select'] = input_data.M_SELECT_LABELS
        labels['camera_x'] = input_data.CAMERA_X_LABELS
        labels['camera_y'] = input_data.CAMERA_Y_LABELS

    self.processed_x_vec = processed_x_vec

    self.head_pdtype = CategoricalPdType(ncat=self.ability_dim)
    self.loc_pdtype = CategoricalPdType(ncat=self.tar_loc_dim)
    self.ptr_pdtype = CategoricalPdType(ncat=self.tar_unit_dim)

    with tf.variable_scope(scope_name, reuse=reuse):
      rl_outputs, il_outputs = self._create_net(x_img=processed_x_img,
                                                x_list=processed_x_list,
                                                x_vec=processed_x_vec,
                                                masks=masks,
                                                labels=labels)
      mha_argmax_tf, mha_tf, neglogp_tf, entropy_tf, vf_tf = rl_outputs
      il_loss_tf = il_outputs[0]
      losses_tf = il_outputs[1]

    self.initial_state = None
    self.il_loss = il_loss_tf
    self.losses = losses_tf

    def step(ob, *_args, **_kwargs):
      """ ob is a tuple of two: one is the state, one is the masks dict """
      state_list, state_vec, state_img, mask_ability, mask_head, \
      mask_select, mask_target, mask_len = ob

      a, v, neglogp = sess.run([mha_tf, vf_tf, neglogp_tf],
                               {X_img: state_img,
                                X_list: state_list,
                                X_vec: state_vec,
                                masks['len']: mask_len,
                                masks['select']: mask_select,
                                masks['target']: mask_target,
                                masks['head']: mask_head,
                                masks['ability']: mask_ability
                                })
      a = [a[0], a[1], a[2], a[3], self._xy_to_loc(a[4], a[5]), a[6], a[7]]
      return a, v, self.initial_state, neglogp

    def value(ob, *_args, **_kwargs):
      state_list, state_vec, state_img, mask_len, \
      mask_select, mask_target, mask_head, mask_ability = ob
      if len(state_img) != 4:
        state_list = np.expand_dims(state_list, axis=0)
        state_vec = np.expand_dims(state_vec, axis=0)
        state_img = np.expand_dims(state_img, axis=0)
        mask_len = np.expand_dims(mask_len, axis=0)
        mask_select = np.expand_dims(mask_select, axis=0)
        mask_target = np.expand_dims(mask_target, axis=0)
        mask_head = np.expand_dims(mask_head, axis=0)
        mask_ability = np.expand_dims(mask_ability, axis=0)

      return sess.run(vf_tf,
                      {X_img: state_img,
                       X_list: state_list,
                       X_vec: state_vec,
                       masks['len']: mask_len,
                       masks['select']: mask_select,
                       masks['target']: mask_target,
                       masks['head']: mask_head,
                       masks['ability']: mask_ability
                       })

    def act(ob):
      state_list, state_vec, state_img, masks_feed = ob

      return sess.run(mha_tf,
                      {X_img: state_img,
                       X_list: state_list,
                       X_vec: state_vec,
                       masks['len']: masks_feed['len'],
                       masks['select']: masks_feed['select'],
                       masks['target']: masks_feed['target'],
                       masks['head']: masks_feed['head'],
                       masks['ability']: masks_feed['ability'],
                       masks['build']: masks_feed['build'],
                       masks['base_pos']: masks_feed['base_pos']
                       })

    self.X_list = X_list
    self.X_vec = X_vec
    self.X_img = X_img
    self.masks = masks
    self.labels = labels
    self.vf = vf_tf
    self.a_argmax = mha_argmax_tf
    self.a = mha_tf
    self.neglogp = neglogp_tf
    self._entropy = entropy_tf
    self.step = step
    self.value = value
    self.il_loss = il_loss_tf
    self.act = act
    PD = namedtuple('pd', ['neglogp', 'entropy'])
    self.pd = PD(self.neglogpac, self.entropy)

  def _create_net(self, x_img, x_list, x_vec, masks, labels):
    ########################### embeddings ###########################
    with tf.variable_scope("camera"):
      cam_embed, _ = self._get_spa_embed(x_img)
    spa_embed, spa_embed_small = self._get_spa_embed(x_img)
    trans_embed, trans_embed_proj, xy = self._get_trans_embed(x_list, masks)
    vec_embed = self._get_vec_embedding(x_vec)
    int_embed = self._integrate_embed(spa_embed_small, trans_embed, vec_embed)
    spa_trans_embed, trans_spa_embed = self._cross_concat(spa_embed,
                                                          trans_embed, xy)

    ########################### main heads ###########################
    # camera
    with tf.variable_scope("camera"):
      camera_heads = self._make_camera_head(cam_embed=cam_embed)
      self.camera_pd = camera_heads.pd
    # ability
    with tf.variable_scope("ability"):
      ab_heads = \
        self._make_general_head(embed=int_embed,
                                n_head=self.ability_dim,
                                mask=masks['ability'])
      self.ab_pd = ab_heads.pd

    # shift
    with tf.variable_scope("shift"):
      sft_heads = \
        self._make_general_head(embed=int_embed, n_head=self.shift_dim)
      self.shift_pd = sft_heads.pd

    # noop_num
    with tf.variable_scope("noop_num"):
      noop_heads = \
        self._make_general_head(embed=int_embed, n_head=self.noop_dim)
      self.noop_pd = noop_heads.pd

    neglogp_final = None
    entropy_final = None
    vf = None
    actions_sam = None
    actions_argmax = None
    final_loss = None
    losses = None
    tar_u_heads = None
    loc_heads = None
    ss_heads = None
    ms_heads = None
    ########################### auto-regressive heads ###########################
    if self.test or self.rl:
      ab_taken = ab_heads.sam
      tar_u_heads, loc_heads, ss_heads, ms_heads = \
        self._create_auto_regressive_part(ab_taken=ab_taken,
                                          tar_xy_taken=None,
                                          masks=masks,
                                          trans_spa_embed=trans_spa_embed,
                                          spa_trans_embed=spa_trans_embed,
                                          tar_u_idx=None,
                                          xy=xy)
      # create final sampled/argmax actions
      actions_sam = [ab_heads.sam, sft_heads.sam, noop_heads.sam, tar_u_heads.sam,
                     loc_heads.sam[0], loc_heads.sam[1], ss_heads.sam, ms_heads.sam,
                     camera_heads.sam[0], camera_heads.sam[1]]
      actions_argmax = [ab_heads.argmax, sft_heads.argmax, noop_heads.argmax,
                        tar_u_heads.argmax, loc_heads.argmax[0], loc_heads.argmax[1],
                        ss_heads.argmax, ms_heads.argmax,
                        camera_heads.argmax[0], camera_heads.argmax[1]]
    if self.rl:
      with tf.variable_scope("RL_value"):
        neglogp_final, entropy_final, vf = \
          self._make_rl_heads(ab_heads, sft_heads, noop_heads, tar_u_heads,
                              loc_heads, ss_heads, ms_heads, camera_heads,
                              int_embed)

    ################### auto-regressive heads with input data ####################
    if self.rl or not self.test:
      ab_taken = tf.cast(labels['ability'], tf.int32)
      tar_xy_taken = tf.concat([
        tf.one_hot(labels['tar_loc_x'], depth=self.map_c_max),
        tf.one_hot(labels['tar_loc_y'], depth=self.map_r_max)
      ], axis=1)
      batch_idx = tf.range(tf.shape(xy[0])[0])
      idx = tf.stack([batch_idx, tf.cast(labels['tar_unit'], tf.int32)], axis=1)
      tar_u_heads, loc_heads, ss_heads, ms_heads = \
        self._create_auto_regressive_part(ab_taken=ab_taken,
                                          tar_xy_taken=tar_xy_taken,
                                          masks=masks,
                                          trans_spa_embed=trans_spa_embed,
                                          spa_trans_embed=spa_trans_embed,
                                          tar_u_idx=idx,
                                          xy=xy)
    if not self.rl:
      with tf.variable_scope("IL_loss"):
        final_loss, losses = self._make_il_loss(ab_heads, sft_heads, noop_heads,
                                                tar_u_heads, loc_heads, ss_heads,
                                                ms_heads, camera_heads, labels)
    self.tar_u_heads, self.loc_heads, self.ss_heads, self.ms_heads = \
      tar_u_heads, loc_heads, ss_heads, ms_heads
    return [actions_argmax, actions_sam, neglogp_final, entropy_final, vf], \
           [final_loss, losses]

  def _make_rl_heads(self, ab_heads, sft_heads, noop_heads, tar_u_heads,
                    loc_heads, ss_heads, ms_heads, camera_heads, int_embed):
    # create final neglogp
    neglogp_final = tf.multiply(
      self.head_mask_selected,
      tf.concat([tf.expand_dims(ab_heads.neglogp, axis=-1),
                 tf.expand_dims(sft_heads.neglogp, axis=-1),
                 tf.expand_dims(noop_heads.neglogp, axis=-1),
                 tf.expand_dims(tar_u_heads.neglogp, axis=-1),
                 tf.expand_dims(loc_heads.neglogp, axis=-1),
                 tf.expand_dims(ss_heads.neglogp, axis=-1),
                 tf.expand_dims(tf.reduce_sum(ms_heads.neglogp,
                                              axis=-1), axis=-1)], axis=-1)
    )
    # create final entropy
    entropy_final = tf.reduce_sum(
      tf.multiply(
        self.head_mask_selected,
        tf.concat([tf.expand_dims(ab_heads.ent, axis=-1),
                   tf.expand_dims(sft_heads.ent, axis=-1),
                   tf.expand_dims(noop_heads.ent, axis=-1),
                   tf.expand_dims(tar_u_heads.ent, axis=-1),
                   tf.expand_dims(loc_heads.ent, axis=-1),
                   tf.expand_dims(ss_heads.ent, axis=-1),
                   tf.expand_dims(tf.reduce_sum(ms_heads.ent,
                                                axis=-1), axis=-1)], axis=-1)
    ), axis=-1)
    # create value
    vf = slim.fully_connected(int_embed, 128, scope='v_fc1')
    vf = slim.fully_connected(vf, 128, scope='v_fc2')
    vf = slim.fully_connected(vf, 1, scope='v_out', activation_fn=None,
                              normalizer_fn=None)
    return neglogp_final, entropy_final, vf

  def _make_il_loss(self, ab_heads, sft_heads, noop_heads,tar_u_heads,
                    loc_heads, ss_heads, ms_heads, camera_heads, labels):
    ab_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels['ability'], logits=ab_heads.logits)
    sft_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels['shift'], logits=sft_heads.logits)
    noop_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels['noop_num'], logits=noop_heads.logits)
    tar_u_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels['tar_unit'], logits=tar_u_heads.logits)
    tar_loc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=self._xy_to_loc(labels['tar_loc_x'], labels['tar_loc_y']),
      logits=loc_heads.logits)
    ss_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels['s_select'], logits=ss_heads.logits)
    ms_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels['m_select'], logits=ms_heads.logits)
    ms_loss = tf.reduce_sum(ms_loss, axis=-1)
    camera_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=self._xy_to_loc(labels['camera_x'], labels['camera_y']),
      logits=camera_heads.logits)

    final_loss = tf.reduce_sum(
      tf.multiply(
        self.head_mask_selected,
        tf.concat([tf.expand_dims(ab_loss, axis=-1),
                   tf.expand_dims(sft_loss, axis=-1),
                   tf.expand_dims(noop_loss, axis=-1),
                   tf.expand_dims(tar_u_loss, axis=-1),
                   tf.expand_dims(tar_loc_loss, axis=-1),
                   tf.expand_dims(ss_loss, axis=-1),
                   tf.expand_dims(ms_loss, axis=-1)], axis=-1)
      ), axis=-1)
    final_loss += camera_loss
    final_loss = tf.reduce_mean(final_loss)

    losses = [tf.multiply(self.head_mask_selected[:, 0], ab_loss),
              tf.multiply(self.head_mask_selected[:, 1], sft_loss),
              tf.multiply(self.head_mask_selected[:, 2], noop_loss),
              tf.multiply(self.head_mask_selected[:, 3], tar_u_loss),
              tf.multiply(self.head_mask_selected[:, 4], tar_loc_loss),
              tf.multiply(self.head_mask_selected[:, 5], ss_loss),
              tf.multiply(self.head_mask_selected[:, 6], ms_loss),
              camera_loss]
    return final_loss, losses

  def _create_auto_regressive_part(self,
                                   ab_taken=None,
                                   tar_xy_taken=None,
                                   masks=None,
                                   trans_spa_embed=None,
                                   spa_trans_embed=None,
                                   tar_u_idx=None,
                                   xy=None):

    # tar_unit
    ab_taken_one_hot = self._process_to_float32(tf.one_hot(ab_taken,
                                                           depth=self.ability_dim))
    with tf.variable_scope("tar_unit"):
      tar_u_head = \
        self._make_ptr_head(
          query=ab_taken_one_hot,
          ptr_mask=tf.gather_nd(
            masks['target'],
            tf.stack([tf.range(tf.shape(ab_taken)[0]),
                      ab_taken], axis=1)),
          memory=trans_spa_embed)

    # tar_loc
    with tf.variable_scope("tar_loc"):
      loc_head = \
        self._make_loc_head(spa_embed=spa_trans_embed,
                            ability_id_one_hot=ab_taken_one_hot,
                            masks=masks)

    # a mass of things
    if tar_xy_taken is None:
      tar_xy_taken = tf.concat([
          tf.one_hot(loc_head.sam[0], depth=self.map_c_max),
          tf.one_hot(loc_head.sam[1], depth=self.map_r_max)
      ], axis=1)

    batch_idx = tf.range(tf.shape(xy[0])[0])
    if tar_u_idx is None:
      tar_u_idx = tf.stack([batch_idx, tar_u_head.sam], axis=1)
    tar_u_xy_taken = tf.concat([
        tf.one_hot(tf.squeeze(tf.gather_nd(xy[0], tar_u_idx), axis=1),
                   depth=self.map_c_max),
        tf.one_hot(tf.squeeze(tf.gather_nd(xy[1], tar_u_idx), axis=1),
                   depth=self.map_r_max),
    ], axis=1)

    head_mask_selected = tf.gather_nd(
      masks['head'],
      tf.stack([tf.range(tf.shape(ab_taken)[0]),
                ab_taken], axis=1))

    self.head_mask_selected = head_mask_selected
    tar_xy_taken = tf.transpose(
        head_mask_selected[:, 3] * tf.transpose(tar_u_xy_taken) + \
        head_mask_selected[:, 4] * tf.transpose(tar_xy_taken),
     )

    ab_x_y_taken = tf.concat([ab_taken_one_hot, tar_xy_taken], axis=1)

    # single_select
    with tf.variable_scope("s_select"):
      ss_head = \
        self._make_ptr_head(query=ab_x_y_taken,
                            ptr_mask=tf.gather_nd(
                              masks['select'],
                              tf.stack([tf.range(tf.shape(ab_taken)[0]),
                                        ab_taken], axis=1)),
                            memory=trans_spa_embed)

    # multi_select
    with tf.variable_scope("m_select"):
      ms_head = \
        self._make_multi_bi_head(memory=trans_spa_embed,
                                 query=ab_x_y_taken,
                                 select_mask=tf.gather_nd(
                                   masks['select'],
                                   tf.stack([tf.range(tf.shape(ab_taken)[0]),
                                             ab_taken], axis=1))
                                 )

    return tar_u_head, loc_head, ss_head, ms_head

  def neglogpac(self, A):
    # TODO: assert A == input_data.A
    A_ab, A_sft, A_noop, A_tar_u, A_loc, A_ss, A_ms = A

    ab_neglogp = self.ab_pd.neglogp(A_ab)
    sft_neglogp = self.shift_pd.neglogp(A_sft)
    noop_neglogp = self.noop_pd.neglogp(A_noop)

    tar_u_neglogp = self.tar_u_heads.pd.neglogp(A_tar_u)
    loc_neglogp = self.loc_heads.pd.neglogp(A_loc)
    ss_neglogp = self.ss_heads.pd.neglogp(A_ss)
    ms_neglogp = self.ms_heads.pd.neglogp(tf.cast(A_ms, tf.int32))

    head_mask_selected_A = tf.reduce_sum(
        tf.multiply(tf.expand_dims(
          tf.cast(tf.one_hot(A_ab, depth=self.ability_dim),
                  tf.float32), axis=-1), self.masks['head']), axis=1)

    a_neglogpac = tf.multiply(
      head_mask_selected_A,
      tf.concat([tf.expand_dims(ab_neglogp, axis=-1),
                 tf.expand_dims(sft_neglogp, axis=-1),
                 tf.expand_dims(noop_neglogp, axis=-1),
                 tf.expand_dims(tar_u_neglogp, axis=-1),
                 tf.expand_dims(loc_neglogp, axis=-1),
                 tf.expand_dims(ss_neglogp, axis=-1),
                 tf.expand_dims(tf.reduce_sum(ms_neglogp,
                                              axis=-1), axis=-1)], axis=-1)
    )

    return a_neglogpac  # [bs, num_heads]

  def entropy(self):
    return self._entropy

  def _get_spa_embed(self, x_img):
    with tf.variable_scope('gridnet_enc'):
      with slim.arg_scope(my_vgg_arg_scope()):
        x = x_img  # NHWC
        x = slim.conv2d(x, 8, [3, 3], scope='conv1')  # [176, 200]
        x = slim.max_pool2d(x, [2, 2], scope='pool1')  # [88, 100]
        x = slim.conv2d(x, 16, [3, 3], scope='conv2')  # [88, 100]
        enc_out = x
        xx_rates = []
        for rate in [3, 6, 12, 24]:  # from Yuan Gao
          xx = slim.conv2d(enc_out, 16, [3, 3], rate=rate,
                           scope=('rate%d_conv1' % rate))
          xx = slim.conv2d(xx, 16, [1, 1], scope=('rate%d_conv2' % rate))
          xx_rates.append(xx)
        x = tf.add_n(xx_rates, name='added_out')
        # resize
        input_size = [tf.shape(x_img)[1], tf.shape(x_img)[2]]
        spatial_embed = tf.image.resize_bilinear(x, input_size, name='out')

        x = slim.max_pool2d(x, [2, 2], scope='pool2')  # [44, 50]
        x = slim.conv2d(x, 8, [3, 3], scope='conv_small1')
        x = slim.max_pool2d(x, [2, 2], scope='pool3')  # [22, 25]
        x = slim.conv2d(x, 4, [5, 5], scope='conv_small2')
        spatial_embed_small = x

        return spatial_embed, spatial_embed_small

  def _get_trans_embed(self, x_list, masks):
    # x_list: [bs, unit_num, dim]
    with tf.variable_scope("trans_enc"):
      x = slim.fully_connected(x_list, self.enc_dim, scope='x_fc1')
      memory = self.trans_encode(x, masks['len'], training=False)
      memory_proj = slim.fully_connected(memory, 4, scope='memory_fc1')

    # 160 is the scalar to normalize the coordinates x, y
    xy = [tf.to_int32(tf.slice(x_list, begin=[0, 0, 0], size=[-1, -1, 1]) * 160),
          tf.to_int32(tf.slice(x_list, begin=[0, 0, 1], size=[-1, -1, 1]) * 160)]
    return memory, memory_proj, xy

  def _get_vec_embedding(self, x_vec):
    with tf.variable_scope("vec_enc"):
      x = slim.fully_connected(x_vec, self.enc_dim, scope='x_fc1')
    return x

  def _cross_concat(self, spatial_embed, trans_embed, xy):
    nbatch = tf.shape(spatial_embed)[0]
    batch_idx = tf.expand_dims(
        tf.tile(tf.expand_dims(tf.range(nbatch), 1), [1, self.tar_unit_dim]), 2)
    idx = tf.concat((batch_idx,) + self._xy_to_rc(xy[0], xy[1]), axis=2)

    spatial_to_trans = tf.gather_nd(spatial_embed, idx)
    # values will be summed up for repeated idx in scatter_nd
    trans_to_spatial = tf.scatter_nd(idx, trans_embed,
                                     [nbatch,
                                      self.map_r_max,
                                      self.map_c_max,
                                      trans_embed.get_shape()[2]])

    # set trans_to_spatial[:, 0, 0, :] = 0
    zz_mask = tf.zeros(shape=[tf.shape(trans_to_spatial)[0],
                              1,
                              tf.shape(trans_to_spatial)[-1]])
    zz_mask = tf.concat([zz_mask,
                         tf.ones(shape=[tf.shape(trans_to_spatial)[0],
                                        tf.shape(trans_to_spatial)[1]-1,
                                        tf.shape(trans_to_spatial)[-1]])],
                        axis=1)
    zz_mask = tf.expand_dims(zz_mask, axis=2)
    zz_mask = tf.concat([zz_mask,
                         tf.ones(shape=[tf.shape(trans_to_spatial)[0],
                                        tf.shape(trans_to_spatial)[1],
                                        tf.shape(trans_to_spatial)[2]-1,
                                        tf.shape(trans_to_spatial)[-1]])],
                        axis=2)
    zz_mask = tf.stop_gradient(zz_mask)
    trans_to_spatial = tf.multiply(trans_to_spatial, zz_mask)

    spatial_trans_embed = tf.concat([spatial_embed, trans_to_spatial], axis=3)
    trans_spatial_embed = tf.concat([trans_embed, spatial_to_trans], axis=2)
    return spatial_trans_embed, trans_spatial_embed

  def _integrate_embed(self, spa_embed, trans_embed, vec_embed):
    """
    :param spa_embed: []
    :param trans_embed: []
    :param vec_embed: []
    :return: vector
    """
    with tf.variable_scope('inte_embed'):
      spa_embed = slim.max_pool2d(spa_embed, [2, 2], scope='pool1')
      spa_embed = slim.conv2d(spa_embed, 16, [3, 3], scope='conv1')
      spa_embed = slim.max_pool2d(spa_embed, [2, 2], scope='pool2')
      spa_embed = slim.conv2d(spa_embed, 16, [3, 3], scope='conv2')
      spa_embed = slim.flatten(spa_embed)
      spa_embed = slim.fully_connected(spa_embed, self.enc_dim, scope='fc1')

      trans_embed = tf.layers.max_pooling1d(trans_embed,
                                            pool_size=self.tar_unit_dim,
                                            strides=1)
      trans_embed = tf.reshape(trans_embed, [-1, self.enc_dim])
      trans_embed = slim.fully_connected(trans_embed, self.enc_dim, scope='fc2')
      inte_embed = tf.concat([spa_embed, trans_embed, vec_embed], axis=-1)

    return inte_embed

  """ heads """
  def _make_camera_head(self, cam_embed):
    head_h = slim.conv2d(cam_embed,
                         1,
                         [1, 1],
                         activation_fn=None,
                         normalizer_fn=None,
                         scope='1x1mapping')  # [bs, r, c, 1]
    head_h = tf.reshape(head_h, [-1, self.map_r_max, self.map_c_max])
    cam_logits_flat = tf.reshape(head_h, [-1, self.map_r_max * self.map_c_max])
    cam_logits_prob = tf.nn.softmax(cam_logits_flat)
    cam_logits_prob = tf.reshape(cam_logits_prob, [-1, self.map_r_max, self.map_c_max])

    cam_argmax_flat = tf.argmax(cam_logits_flat, axis=-1)
    cam_argmax_x, cam_argmax_y = self._loc_to_xy(cam_argmax_flat)

    cam_pd = self.loc_pdtype.pdfromflat(cam_logits_flat)
    cam_sam_flat = cam_pd.sample()
    cam_sam_x, cam_sam_y = self._loc_to_xy(cam_sam_flat)

    cam_neglogp_flat = cam_pd.neglogp(cam_sam_flat)
    cam_entropy = cam_pd.entropy()

    return Head(cam_logits_flat, [cam_argmax_x, cam_argmax_y], \
                [cam_sam_x, cam_sam_y], cam_neglogp_flat, cam_pd, cam_entropy)

  def _make_general_head(self, embed, n_head, mask=None):
    """
    :param embed: [None, num_units, encoding dim]
    :param n_head: output dim
    :param pd_type:
    :return:
    """
    head_h = embed
    head_h = slim.fully_connected(head_h, self.enc_dim, scope='fc1')
    head_logits = slim.fully_connected(head_h,
                                       n_head,
                                       scope='logits',
                                       activation_fn=None,
                                       normalizer_fn=None)
    # hack build
    if n_head == self.ability_dim:
      zero_offsets = tf.zeros(shape=(tf.shape(embed)[0], 1))
      ones_offsets = tf.ones(shape=(tf.shape(embed)[0], 1))  # e^1, 2.7 times
      overall_offsets = tf.concat([tf.tile(zero_offsets, [1, 15]),
                                   ones_offsets - 1.0,  # hatchery
                                   ones_offsets + 2.0,  # hydraliskden
                                   ones_offsets,  # infestationpit
                                   tf.tile(zero_offsets, [1, 2]),
                                   ones_offsets,  # roachwarren
                                   tf.tile(zero_offsets, [1, 4]),
                                   ones_offsets,  # ultraliskcavern
                                   tf.tile(zero_offsets, [1, 43]),
                                   ones_offsets + 2.0,  # morphlair
                                   tf.tile(zero_offsets, [1, 67])], axis=1)
      head_logits += overall_offsets

    if mask is not None:
      head_logits = self._mask_logits(head_logits, mask)

    head_argmax = tf.argmax(head_logits, axis=-1)
    head_pd = self.head_pdtype.pdfromflat(head_logits)
    head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    head_entropy = head_pd.entropy()

    return Head(head_logits, head_argmax, head_sam, head_neglogp, head_pd, head_entropy)

  def _make_multi_bi_head(self, memory, query, select_mask):
    # action embedding
    query_h = slim.fully_connected(query, self.enc_dim, scope='query_fc1')
    """ memory: [None, unit_max_num, enc_dim] """
    query_h = tf.expand_dims(query_h, axis=1)
    query_h = tf.tile(query_h, multiples=[1, tf.shape(memory)[1], 1])
    head_h = tf.concat([memory, query_h], axis=-1)
    head_h = slim.fully_connected(head_h, self.enc_dim, scope='fc1')
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
    head_entropy = head_pd.entropy()

    return Head(head_logits, head_argmax, head_sam, head_neglogp, head_pd, head_entropy)

  def _make_ptr_head(self, query, ptr_mask, memory):
    # action embedding
    query_h = slim.fully_connected(query, self.enc_dim, scope='query_fc1')
    query_h = tf.expand_dims(query_h, axis=1)

    select_logits, select_prob = self.trans_decode(query_h, memory, training=False)
    select_logits = tf.reshape(select_logits, [-1, self.tar_unit_dim])

    # modify the logits that unavailable position will be -inf
    neginf = tf.zeros_like(select_logits) - INF
    _mask = tf.equal(ptr_mask, 0)
    select_logits = tf.where(_mask, neginf, select_logits)

    select_argmax = tf.argmax(select_logits, axis=-1)
    select_pd = self.head_pdtype.pdfromflat(select_logits)
    select_sam = select_pd.sample()
    select_neglogp = select_pd.neglogp(select_sam)
    select_entropy = select_pd.entropy()

    return Head(select_logits, select_argmax, select_sam, select_neglogp, select_pd, select_entropy)

  def _make_loc_head(self, spa_embed, ability_id_one_hot, masks):
    loc_masks = self._make_loc_head_masks(masks)
    head_h = slim.conv2d(spa_embed,
                         self.ability_dim,
                         [1, 1],
                         activation_fn=None,
                         normalizer_fn=None,
                         scope='1x1mapping')  # [bs, r, c, 137]
    loc_logits = head_h
    loc_logits = tf.transpose(loc_logits, perm=[0, 3, 1, 2])  # [bs, 137, r, c]

    ability_id_one_hot = tf.expand_dims(ability_id_one_hot, axis=-1)
    ability_id_one_hot = tf.expand_dims(ability_id_one_hot, axis=-1)

    loc_logits = tf.multiply(loc_logits, ability_id_one_hot)
    loc_logits = tf.reduce_sum(loc_logits, axis=1)  # [bs, r, c]

    loc_masks = tf.multiply(loc_masks, ability_id_one_hot)
    loc_masks = tf.reduce_sum(loc_masks, axis=1)  # [bs, r, c]

    loc_logits = self._mask_logits(logits=loc_logits, mask=loc_masks)

    loc_logits_flat = tf.reshape(loc_logits, [-1, self.map_r_max * self.map_c_max])
    loc_logits_prob = tf.nn.softmax(loc_logits_flat)
    loc_logits_prob = tf.reshape(loc_logits_prob, [-1, self.map_r_max, self.map_c_max])

    loc_argmax_flat = tf.argmax(loc_logits_flat, axis=-1)
    loc_argmax_x, loc_argmax_y = self._loc_to_xy(loc_argmax_flat)

    loc_pd = self.loc_pdtype.pdfromflat(loc_logits_flat)
    loc_sam_flat = loc_pd.sample()
    loc_sam_x, loc_sam_y = self._loc_to_xy(loc_sam_flat)

    loc_neglogp_flat = loc_pd.neglogp(loc_sam_flat)
    loc_entropy = loc_pd.entropy()

    return Head(loc_logits_flat, [loc_argmax_x, loc_argmax_y], \
                [loc_sam_x, loc_sam_y], loc_neglogp_flat, loc_pd, loc_entropy)

  def _make_loc_head_masks(self, masks):
    # self.masks['build']  # [bs, map_r, map_c]
    ones_mask = tf.ones_like(masks['build'])
    # self.masks['base_pos']  # [bs, 16, 2]
    base_pos_x = tf.slice(masks['base_pos'], [0, 0, 0], [-1, -1, 1])
    base_pos_x = tf.reshape(base_pos_x, [-1, self.max_bases_num])
    base_pos_y = tf.slice(masks['base_pos'], [0, 0, 1], [-1, -1, 1])
    base_pos_y = tf.reshape(base_pos_y, [-1, self.max_bases_num])
    base_pos_r, base_pos_c = self._xy_to_rc(base_pos_x, base_pos_y)
    base_rc = tf.stack([base_pos_r, base_pos_c], axis=2)

    nbatch = tf.shape(masks['base_pos'])[0]
    batch_idx = tf.expand_dims(
        tf.tile(tf.expand_dims(tf.range(nbatch), 1), [1, self.max_bases_num]), 2)
    idx = tf.concat([batch_idx, tf.cast(base_rc, tf.int32)], axis=2)
    # values will be summed up for repeated idx in scatter_nd
    # [bs, 176, 200]
    base_mask = tf.scatter_nd(idx, tf.ones(shape=(nbatch, self.max_bases_num)),
                              [nbatch, self.map_r_max, self.map_c_max])
    # 0-9 ones_mask; 10-13 build_mask; 14 ones_mask; 15 base_mask;
    # 16-25 build_mask; 26-136 ones_mask
    ones_mask = tf.expand_dims(ones_mask, axis=1)  # [bs, 1, map_r, map_c]
    base_mask = tf.expand_dims(base_mask, axis=1)  # [bs, 1, map_r, map_c]
    build_mask = tf.expand_dims(masks['build'], axis=1)  # [bs, 1, map_r, map_c]
    # [bs, 137, map_r, map_c]
    loc_mask = tf.concat([tf.tile(ones_mask, [1, 10, 1, 1]),
                          tf.tile(build_mask, [1, 4, 1, 1]),
                          ones_mask,
                          base_mask,
                          tf.tile(build_mask, [1, 10, 1, 1]),
                          tf.tile(ones_mask, [1, 111, 1, 1])], axis=1)
    return loc_mask

  def trans_encode(self, x, len_mask, training=True):
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

  def trans_decode(self, y, memory, training=True):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      dec_logits, dec_pd = [], []
      dec = y
      # Blocks
      for i in range(self.num_dec_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
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

  def _process_to_float32(self, mask):
    return tf.cast(mask, tf.float32)

  def _mask_logits(self, logits, mask):
    neginf = tf.zeros_like(logits) - INF
    _mask = tf.equal(mask, 0)
    logits = tf.where(_mask, neginf, logits)
    return logits

  def _loc_to_xy(self, loc):
    r, c = self._loc_to_rc(loc)
    x, y = self._rc_to_xy(r, c)
    return x, y

  def _xy_to_loc(self, x, y):
    r, c = self._xy_to_rc(x, y)
    loc = self._rc_to_loc(r, c)
    return loc

  def _loc_to_rc(self, loc):
    r = loc // self.map_c_max
    c = loc % self.map_c_max
    return r, c

  def _rc_to_loc(self, r, c):
    loc = r * self.map_c_max + c
    return loc

  def _xy_to_rc(self, x, y):
    r = self.map_r_max - 1 - y
    c = x
    return r, c

  def _rc_to_xy(self, r, c):
    x = c
    y = self.map_r_max - 1 - r
    return x, y


def transformer_test():
  nbatch = 8

  unit_num_max = 600
  feat_dim = 128
  img_size = (176, 200, 10)
  vec_dim = 10

  ob_list_space = spaces.Box(low=0, high=1, shape=(unit_num_max, feat_dim),
                             dtype=float)
  ob_vec_space = spaces.Box(low=0, high=1, shape=(vec_dim,), dtype=float)
  ob_img_space = spaces.Box(low=0, high=1, shape=img_size, dtype=float)
  ob_space = spaces.Tuple([ob_list_space, ob_vec_space, ob_img_space])

  # from timitate.lib.pb2action_converter import PB2ActionConverter
  # ac_space = PB2ActionConverter().space
  ac_space = spaces.Tuple([spaces.Discrete(137),
                           spaces.Discrete(2),
                           spaces.Discrete(10),
                           spaces.Discrete(unit_num_max),
                           spaces.Discrete(200),
                           spaces.Discrete(176),
                           spaces.Discrete(unit_num_max),
                           spaces.Box(low=0, high=1,
                                      shape=(unit_num_max,),
                                      dtype=np.bool),
                           spaces.Discrete(200),
                           spaces.Discrete(176)
                           ])

  sess = tf.Session()
  model = SpatialTrans(sess=sess, ob_space=ob_space, ac_space=ac_space,
                       test=True, rl=False)
  print('Successfully build the model.')

  tf.global_variables_initializer().run(session=sess)
  import joblib
  params = tf.trainable_variables()
  # for p in params:
  #   print(p)
  ps = sess.run(params)
  joblib.dump(ps, 'spatial_trans.ckpt')

  state_img = np.zeros((nbatch,) + img_size)
  state_list = np.zeros([nbatch, unit_num_max, feat_dim])
  state_vec = np.zeros([nbatch, vec_dim])

  masks_feed = {}
  masks_feed['len'] = np.zeros([nbatch, unit_num_max])
  masks_feed['select'] = np.zeros([nbatch, 137, unit_num_max])
  masks_feed['target'] = np.zeros([nbatch, 137, unit_num_max])
  masks_feed['head'] = np.zeros([nbatch, 137, 7])
  masks_feed['ability'] = np.zeros([nbatch, 137])
  masks_feed['build'] = np.zeros([nbatch, 176, 200])
  masks_feed['base_pos'] = np.zeros([nbatch, 16, 2])  # 16 is the num of bases

  ob = [state_list, state_vec, state_img, masks_feed]
  mha = model.act(ob)

  # train
  labels = {}
  labels['ability'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['shift'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['noop_num'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['tar_unit'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['tar_loc_x'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['tar_loc_y'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['s_select'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['m_select'] = np.zeros(shape=(nbatch, unit_num_max), dtype=np.int32)
  labels['camera_x'] = np.zeros(shape=(nbatch,), dtype=np.int32)
  labels['camera_y'] = np.zeros(shape=(nbatch,), dtype=np.int32)

  loss = model.il_loss
  trainable_params = tf.trainable_variables()
  trainer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
  grads = tf.gradients(loss, trainable_params)
  grads_v = list(zip(grads, trainable_params))
  train_op = trainer.apply_gradients(grads_v)
  tf.global_variables_initializer().run(session=sess)

  train_step = 0
  import time
  total_time = 0
  while True:
    t1 = time.time()
    _, il_loss, losses = sess.run([train_op, model.il_loss, model.losses], {model.X_img: state_img,
                        model.X_list: state_list,
                        model.X_vec: state_vec,
                        model.masks['len']: masks_feed['len'],
                        model.masks['select']: masks_feed['select'],
                        model.masks['target']: masks_feed['target'],
                        model.masks['head']: masks_feed['head'],
                        model.masks['ability']: masks_feed['ability'],
                        model.masks['build']: masks_feed['build'],
                        model.masks['base_pos']: masks_feed['base_pos'],
                        model.labels['ability']: labels['ability'],
                        model.labels['shift']: labels['shift'],
                        model.labels['noop_num']: labels['noop_num'],
                        model.labels['tar_unit']: labels['tar_unit'],
                        model.labels['tar_loc_x']: labels['tar_loc_x'],
                        model.labels['tar_loc_y']: labels['tar_loc_y'],
                        model.labels['s_select']: labels['s_select'],
                        model.labels['m_select']: labels['m_select'],
                        model.labels['camera_x']: labels['camera_x'],
                        model.labels['camera_y']: labels['camera_y']
                        })
    # print('total loss: {}, losses: {}'.format(il_loss, losses))
    t2 = time.time()
    total_time += t2 - t1
    train_step += 1
    print('batch: {}, fps: {}'.format(train_step, train_step * float(nbatch) / total_time))


if __name__ == '__main__':
  transformer_test()
