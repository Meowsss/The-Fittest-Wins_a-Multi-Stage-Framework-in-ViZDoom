import horovod.tensorflow as hvd
import tensorflow as tf

from tleague.learners.ppo_learner2 import PPOLearner
from tleague.learners.data_server import DataServer
from tleague.utils import logger
from tleague.utils.data_structure import PPOData


class HvdPPOLearner(PPOLearner):

  def __init__(self,league_mgr_addr, model_pool_addrs, learner_ports,
               rm_size, batch_size, ob_space, ac_space, policy, gpu_id,
               policy_config={}, ent_coef=1e-2, distill_coef=1e-2,
               vf_coef=0.5, max_grad_norm=0.5, rwd_shape=False,
               pub_interval=500, log_interval=100, save_interval=0,
               total_timesteps=5e7, burn_in_timesteps=0,
               learner_id='', batch_worker_num=4, pull_worker_num=2,
               unroll_length=32, rollout_length=1,
               use_mixed_precision=False, use_sparse_as_dense=True,
               adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-5,
               ep_loss_coef={}, **kwargs):
    super(PPOLearner, self).__init__(league_mgr_addr, model_pool_addrs,
                                     learner_ports, learner_id)

    self._init_const(total_timesteps, burn_in_timesteps, batch_size,
                     unroll_length, rwd_shape, ent_coef, vf_coef,
                     pub_interval, log_interval, save_interval,
                     policy, distill_coef, policy_config, rollout_length,
                     ep_loss_coef)

    # allow_soft_placement=True can fix issue when some op cannot be defined on
    # GPUs for tf-1.8.0; tf-1.13.1 does not have this issue
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)
    self.sess = tf.Session(config=config)

    ## Prepare dataset
    ds = PPOData(ob_space, ac_space, self.n_v, use_lstm=self.rnn, hs_len=self.hs_len,
                 distillation=self.distillation, version='v2')
    self._data_server = DataServer(self._pull_data, rm_size,
                                   unroll_length, batch_size, ds,
                                   gpu_id_list=(0,),
                                   batch_worker_num=batch_worker_num,
                                   pull_worker_num=pull_worker_num,
                                   rollout_length=rollout_length,
                                   prefetch_buffer_size=2)

    net_config = policy.net_config_cls(ob_space, ac_space, **policy_config)
    net_config.clip_range = self.CLIPRANGE
    if rwd_shape:
      ## NOTE： Assume there is reward_weights_shape in net_config
      reward_weights_shape = net_config.reward_weights_shape
      self.rwd_weights = tf.placeholder(tf.float32, reward_weights_shape)
      net_config.reward_weights = self.rwd_weights

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as model_scope:
      pass
    def create_policy(inputs, nc):
      return policy.net_build_fun(inputs=inputs, nc=nc, scope=model_scope)

    device = '/gpu:{}'.format(0)
    with tf.device(device):
      input_data = self._data_server.input_datas[0]
      if 'use_xla' in policy_config and policy_config['use_xla']:
        try:
          # Use tensorflow's accerlated linear algebra compile method
          with tf.xla.experimental.jit_scope(True):
            model = create_policy(input_data, net_config)
        except:
          logger.log("WARNING: using tf.xla requires tf version>=1.15.")
          model = create_policy(input_data, net_config)
      else:
        model = create_policy(input_data, net_config)
      loss, vf_loss, losses = self.build_loss(model, input_data)
    self.losses = [hvd.allreduce(loss) for loss in losses]
    self.params = tf.trainable_variables(scope='model')
    self.params_vf = tf.trainable_variables(scope='model/vf')
    self.param_norm = tf.global_norm(self.params)

    trainer = tf.train.AdamOptimizer(learning_rate=self.LR,
                                     beta1=adam_beta1,
                                     beta2=adam_beta2,
                                     epsilon=adam_eps)
    burn_in_trainer = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)  # same as default and IL
    if use_mixed_precision:
      try:
        self.trainer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(self.trainer)
        self.burn_in_trainer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(self.burn_in_trainer)
      except:
        logger.warn("using tf mixed_precision requires tf version>=1.15.")
    self.trainer = hvd.DistributedOptimizer(trainer, sparse_as_dense=use_sparse_as_dense)
    self.burn_in_trainer = hvd.DistributedOptimizer(burn_in_trainer, sparse_as_dense=use_sparse_as_dense)
    grads_and_vars = self.trainer.compute_gradients(loss, self.params)
    grads_and_vars_vf = self.burn_in_trainer.compute_gradients(vf_loss, self.params_vf)
    clip_vars = model.vars.lstm_vars
    grads_and_vars, self.clip_grad_norm, self.nonclip_grad_norm = self.clip_grads_vars(
      grads_and_vars, clip_vars, max_grad_norm)
    grads_and_vars_vf, self.clip_grad_norm_vf, self.nonclip_grad_norm_vf = self.clip_grads_vars(
      grads_and_vars_vf, clip_vars, max_grad_norm)

    self._train_batch = self.trainer.apply_gradients(grads_and_vars)
    self._burn_in = self.burn_in_trainer.apply_gradients(grads_and_vars_vf)
    self.loss_endpoints_names = model.loss.loss_endpoints.keys()
    self._build_ops()
    barrier_op = hvd.allreduce(tf.Variable(0.))
    broadcast_op = hvd.broadcast_global_variables(0)
    tf.global_variables_initializer().run(session=self.sess)
    self.sess.graph.finalize()

    self.barrier = lambda : self.sess.run(barrier_op)
    self.broadcast = lambda : self.sess.run(broadcast_op)
    self.broadcast()
    # logging stuff
    format_strs = (['stdout', 'log', 'tensorboard', 'csv'] if hvd.rank() == 0
                   else ['stdout', 'log', 'tensorboard', 'csv'])
    logger.configure(
      dir='training_log/{}rank{}'.format(self._learner_id, hvd.rank()),
      format_strs=format_strs
    )

  def run(self):
    logger.log('HvdPPOLearner: entering run()')
    first_lp = True
    while True:
      if hvd.rank() == 0:
        if first_lp:
          first_lp = False
          task = self._query_task()
          if task is not None:
            self.task = task
          else:
            self.task = self._request_task()
        else:
          self.task = self._request_task()
        logger.log('rank{}: done _request_task'.format(
          hvd.rank()))
      self.barrier()
      if hvd.rank() == 0:
        self._init_task()
        self._notify_task_begin(self.task)
        logger.log('rank{}: done init_task and notify_task_begin...'.format(
          hvd.rank()))
      else:
        self.task = self._query_task()
        logger.log('rank{}: done _query_task...'.format(hvd.rank()))
      self.barrier()

      logger.log('rank{}: broadcasting...'.format(hvd.rank()))
      self.broadcast()
      logger.log('rank{}: done broadcasting'.format(hvd.rank()))
      self._train()
      self._lrn_period_count += 1

  def _train(self, **kwargs):
    self._data_server._update_model_id(self.model_key)
    # Use different model, clear the replay memory
    if self.last_model_key is None or self.last_model_key != self.task.parent_model_key:
      self._data_server.reset()
      if self._lrn_period_count == 0:
        self._need_burn_in = self.burn_in_timesteps > 0
      else:
        self._need_burn_in = True
    else:
      self._need_burn_in = False

    self.barrier()
    nbatch = self.batch_size * hvd.size()
    self.should_push_model = (hvd.rank() == 0)
    self._run_train_loop(nbatch)
