import horovod.tensorflow as hvd
import tensorflow as tf
from tleague.learners.data_server import DataServer
from tleague.learners.ppo_learner import PPOLearner
from tleague.utils import logger
from tleague.utils.data_structure import PPOData


class HvdPPOLearner(PPOLearner):

  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports,
               rm_size, batch_size, ob_space, ac_space, policy, gpu_id,
               policy_config={}, ent_coef=1e-2, unroll_length=32,
               n_v=1, vf_coef=0.5, max_grad_norm=0.5,
               pub_interval=500, log_interval=100, save_interval=0,
               total_timesteps=5e7, burn_in_timesteps=0,
               adv_normalize=True, rwd_shape=False, merge_pi=False,
               learner_id='', batch_worker_num=4, pull_worker_num=2,
               distill_coef=1e-2, rollout_length=1, **kwargs):
    super(PPOLearner, self).__init__(league_mgr_addr, model_pool_addrs,
                                     learner_ports, learner_id)

    self._init_const(total_timesteps, burn_in_timesteps, batch_size,
                     unroll_length, ob_space, ac_space, rwd_shape,
                     merge_pi, n_v, adv_normalize, ent_coef, vf_coef,
                     pub_interval, log_interval, save_interval,
                     policy, policy_config, distill_coef)

    # allow_soft_placement=True can fix issue when some op cannot be defined on
    # GPUs for tf-1.8.0; tf-1.13.1 does not have this issue
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(gpu_id)
    self.sess = tf.Session(config=config)
    if self.rnn:
      policy_config['rollout_len'] = rollout_length

    ## Prepare dataset
    ds = PPOData(ob_space, ac_space, n_v, use_lstm=self.rnn, hs_len=self.hs_len,
                 distillation=self.distillation, version='v1')
    self._data_server = DataServer(self._pull_data, rm_size,
                                   unroll_length, batch_size, ds,
                                   gpu_id_list=(0,),
                                   batch_worker_num=batch_worker_num,
                                   pull_worker_num=pull_worker_num,
                                   rollout_length=rollout_length,
                                   prefetch_buffer_size=2)

    device = '/gpu:{}'.format(0)
    with tf.device(device):
      input_data = self._data_server.input_datas[0]
      model = policy(ob_space=ob_space, ac_space=ac_space, nbatch=batch_size,
                     n_v=n_v, input_data=input_data, reuse=tf.AUTO_REUSE,
                     scope_name='model', **policy_config)
      loss, vf_loss, self.losses = self.build_loss(model, input_data)
    self.params = tf.trainable_variables(scope='model')
    self.params_vf = tf.trainable_variables(scope='model/vf')
    self.param_norm = tf.global_norm(self.params)

    trainer = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
    burn_in_trainer = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
    self.trainer = hvd.DistributedOptimizer(trainer, sparse_as_dense=True)
    self.burn_in_trainer = hvd.DistributedOptimizer(burn_in_trainer, sparse_as_dense=True)
    grads_and_vars = self.trainer.compute_gradients(loss, self.params)
    grads = [grad for grad, var in grads_and_vars]
    tvars = [var for grad, var in grads_and_vars]
    grads_and_vars_vf = self.burn_in_trainer.compute_gradients(vf_loss, self.params_vf)
    grads_vf = [grad for grad, var in grads_and_vars_vf]
    tvars_vf = [var for grad, var in grads_and_vars_vf]
    if max_grad_norm is not None:
      grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
      grads_vf, self.grad_vf_norm = tf.clip_by_global_norm(grads_vf, max_grad_norm)
    else:
      self.grad_norm = tf.global_norm(grads)
      self.grad_vf_norm = tf.global_norm(grads_vf)
    self._train_batch = self.trainer.apply_gradients(zip(grads, tvars))
    self._burn_in = self.burn_in_trainer.apply_gradients(zip(grads_vf, tvars_vf))

    self._build_ops()
    barrier_op = hvd.allreduce(tf.Variable(0.))
    broadcast_op = hvd.broadcast_global_variables(0)
    tf.global_variables_initializer().run(session=self.sess)
    self.sess

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
    while True:
      if hvd.rank() == 0:
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

  def _query_task(self):
    task = self._league_mgr_apis.query_learner_task(self._learner_id)
    self.last_model_key = self.model_key
    self.model_key = task.model_key
    return task

  def _train(self, **kwargs):
    self._data_server._update_model_id(self.model_key)
    # Use different model, clear the replay memory
    if self.last_model_key is None or self.last_model_key != self.task.parent_model_key:
      self._data_server.reset()
    self.barrier()
    nbatch = self.batch_size * hvd.size()
    self.should_push_model = (hvd.rank() == 0)
    self._run_train_loop(nbatch)
