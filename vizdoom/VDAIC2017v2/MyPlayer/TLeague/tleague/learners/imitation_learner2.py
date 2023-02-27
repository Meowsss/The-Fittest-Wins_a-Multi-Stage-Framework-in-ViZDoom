""" Imitation Learner2, combines and is based on
ImitationLearner & HvdImitationLearner """
import time

import tensorflow as tf
try:
  import horovod.tensorflow as hvd
  from tleague.learners.piecewise_fusion import pw_fusion
  has_hvd = True
except:
  has_hvd = False

from tleague.learners.data_server import ImDataServer
from tleague.model_pools.model_pool_apis import ModelPoolAPIs
from tleague.utils import logger
from tleague.utils.chkpts import ChkptsFromSelf


def _get_local_replays(replay_filelist):
  import csv
  from tleague.utils.tl_types import ImitationTask
  train_replay_filelist = []
  val_replay_filelist = []
  with open(replay_filelist, "r") as f:
    reader = csv.DictReader(f, delimiter=",")
    for i, line in enumerate(reader):
      if not has_hvd or i % hvd.size() == hvd.rank():
        if int(line['validation']):
          val_replay_filelist.append(ImitationTask(**line))
        else:
          train_replay_filelist.append(ImitationTask(**line))
  return train_replay_filelist, val_replay_filelist


class ImitationLearner2(object):
  """ Imitation Learner2.

  Support either CPU or GPU (with or without Horovod).
  """

  def __init__(self, ports, gpu_id, replay_filelist, batch_size,
               min_train_sample_num, min_val_sample_num, rm_size,
               learning_rate, print_interval, checkpoint_interval,
               num_val_batches, replay_converter_type, policy,
               policy_config=None, max_clip_grad_norm=1,
               checkpoints_dir=None, restore_checkpoint_path=None,
               train_generator_worker_num=4, val_generator_worker_num=2,
               pull_worker_num=2, num_sgd_updates=int(1e30),
               repeat_training_task=False, unroll_length=32,
               rollout_length=1, model_pool_addrs=None,
               pub_interval=50, converter_config=None, l2_coef=0.001,
               use_mixed_precision=False, pw_fusion_schedule: str = "-1",
               use_sparse_as_dense=True):
    assert len(ports) == 2
    self.rank = 0 if not has_hvd else hvd.rank()
    self.model_key = 'IL-model'
    self.pub_interval = pub_interval
    policy_config = {} if policy_config is None else policy_config
    self.rnn = (False if 'use_lstm' not in policy_config
                else policy_config['use_lstm'])
    self.hs_len = None
    if self.rnn:
      assert model_pool_addrs is not None
      self._model_pool_apis = ModelPoolAPIs(model_pool_addrs)
      self._model_pool_apis.check_server_set_up()
      policy_config['rollout_len'] = rollout_length
      self.hs_len = (policy_config['hs_len'] if ('hs_len'in policy_config)
                     else 2 * policy_config['nlstm'] if ('nlstm' in policy_config) else 128)
    self.should_push_model = (self.rnn and self.rank == 0)
    use_gpu = (gpu_id >= 0)
    converter_config = {} if converter_config is None else converter_config
    train_replay_filelist, val_replay_filelist = _get_local_replays(replay_filelist)
    replay_converter = replay_converter_type(**converter_config)
    ob_space, ac_space = replay_converter.space.spaces
    self.data_pool = ImDataServer(
      ports=ports,
      train_replay_filelist=train_replay_filelist,
      val_replay_filelist=val_replay_filelist,
      batch_size=batch_size,
      min_train_sample_num=min_train_sample_num,
      min_val_sample_num=min_val_sample_num,
      ob_space=ob_space,
      ac_space=ac_space,
      train_generator_worker_num=train_generator_worker_num,
      val_generator_worker_num=val_generator_worker_num,
      pull_worker_num=pull_worker_num,
      rm_size=rm_size,
      repeat_training_task=repeat_training_task,
      unroll_length=unroll_length,
      rollout_length=rollout_length,
      lstm=self.rnn,
      hs_len=self.hs_len,
      use_gpu=use_gpu
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    if use_gpu:
      config.gpu_options.visible_device_list = str(gpu_id)
      config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)

    device = '/gpu:0' if use_gpu else '/cpu:0'
    def create_policy():
      return policy(sess=self._sess,
                    ob_space=ob_space,
                    ac_space=ac_space,
                    nbatch=batch_size,
                    input_data=self.data_pool.train_batch_input,
                    **policy_config)
    with tf.device(device):
      if 'use_xla' in policy_config and policy_config['use_xla']:
        try:
          # Use tensorflow's accerlated linear algebra compile method
          with tf.xla.experimental.jit_scope(True):
            model = create_policy()
        except:
          logger.log("WARNING: using tf.xla in im-lrn2 requires tf version>=1.15.")
          model = create_policy()
      else:
        model = create_policy()

    model_val = policy(sess=self._sess,
                       ob_space=ob_space,
                       ac_space=ac_space,
                       nbatch=batch_size,
                       input_data=self.data_pool.val_batch_input,
                       reuse=True,
                       **policy_config)
    params = tf.trainable_variables()
    param_norm = tf.global_norm(params)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       epsilon=1e-5)
    if use_mixed_precision:
      try:
        optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
      except:
        logger.log("WARNING: using tf mixed_precision requires tf version>=1.15.")
    if has_hvd:
      if pw_fusion_schedule is None:
        optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=use_sparse_as_dense)
      barrier_op = hvd.allreduce(tf.Variable(0.))
      self.barrier = lambda: self._sess.run(barrier_op)
    train_loss = tf.reduce_mean(
      model.il_loss * self.data_pool.train_batch_weight
    )
    val_loss = tf.reduce_mean(
      model_val.il_loss * self.data_pool.val_batch_weight
    )
    total_loss = train_loss + l2_coef * tf.square(param_norm)
    grads_and_vars = optimizer.compute_gradients(total_loss, params)
    if pw_fusion_schedule is not None:
      grads_and_vars = pw_fusion(pw_fusion_schedule, grads_and_vars, batch_size,
                                 sparse_as_dense=use_sparse_as_dense)
    clip_vars = model.get_lstm_vars()
    clip_grads = [grad for grad, var in grads_and_vars if var in clip_vars]
    nonclip_grads_and_vars = [(grad, var) for grad, var in grads_and_vars if var not in clip_vars]
    if max_clip_grad_norm > 0:
      clip_grads, clip_grad_norm = tf.clip_by_global_norm(clip_grads, max_clip_grad_norm)
    else:
      clip_grad_norm = tf.global_norm(clip_grads)
    clip_grads_and_var = list(zip(clip_grads, clip_vars))
    grads_and_vars = clip_grads_and_var + nonclip_grads_and_vars
    grad_norm = tf.global_norm(list(zip(*grads_and_vars))[0])

    train_op = optimizer.apply_gradients(grads_and_vars)
    tf.global_variables_initializer().run(session=self._sess)

    self.new_params = [tf.placeholder(p.dtype, shape=p.get_shape())
                       for p in params]
    self.param_assign_ops = [p.assign(new_p)
                             for p, new_p in zip(params, self.new_params)]
    opt_params = optimizer.variables()
    self.new_opt_params = [tf.placeholder(p.dtype, shape=p.get_shape())
                           for p in opt_params]
    self.opt_param_assign_ops = [
      p.assign(new_p) for p, new_p in zip(opt_params, self.new_opt_params)
    ]

    def read_params():
      return self._sess.run(params)

    def read_opt_params():
      return self._sess.run(opt_params)

    def load_model(loaded_params):
      self._sess.run(self.param_assign_ops, feed_dict={
        p: v for p, v in zip(self.new_params, loaded_params)
      })

    def restore_optimizer(loaded_opt_params):
      self._sess.run(self.opt_param_assign_ops, feed_dict={
        p: v for p, v in zip(self.new_opt_params, loaded_opt_params)
      })

    def _train_step():
      return self._sess.run([train_loss_aggregated, *train_head_loss_aggregated,
                             grad_norm, clip_grad_norm, param_norm, train_op], {})[:-1]
    # maximal_feat = [tf.reduce_max(tf.cast(x, tf.float32)) for x in self.data_pool.val_batch_input.X]
    def _val_step():
      # print(self._sess.run(maximal_feat, {}))
      return self._sess.run(
          [val_loss_aggregated, *val_head_losses_aggregated], {})

    self._saver = ChkptsFromSelf(read_params, load_model, self.model_key)

    if restore_checkpoint_path is not None:
      self._saver._restore_model_checkpoint(restore_checkpoint_path)

    if has_hvd:
      hvd.broadcast_global_variables(0).run(session=self._sess)

    _allreduce = lambda x: x if not has_hvd else hvd.allreduce(x)
    train_loss_aggregated = _allreduce(train_loss)
    train_head_loss_aggregated = [
      _allreduce(tf.reduce_mean(l * self.data_pool.train_batch_weight))
      for l in model.losses
    ]
    val_loss_aggregated = _allreduce(val_loss)
    val_head_losses_aggregated = [
      _allreduce(tf.reduce_mean(l * self.data_pool.val_batch_weight))
      for l in model_val.losses
    ]
    self._sess.graph.finalize()
    self._total_samples = lambda: [self.data_pool._num_train_samples,
                                   self.data_pool._num_val_samples]
    self._names = ['loss'] + model.loss_names + ['grad_norm', 'clip_grad_norm', 'param_norm']
    self._batch_size = batch_size
    self._train_step = _train_step
    self._val_step = _val_step
    self._print_interval = print_interval
    self._checkpoint_interval = checkpoint_interval
    self._num_val_batches = num_val_batches
    self._checkpoints_dir = checkpoints_dir if self.rank == 0 else None
    self._num_sgd_updates = num_sgd_updates
    self.load_model = load_model
    self.restore_optimizer = restore_optimizer
    self.read_params = read_params
    self.read_opt_params = read_opt_params

    format_strs = ['stdout', 'log', 'tensorboard', 'csv']
    logger.configure(
        dir='training_log/rank{}'.format(self.rank),
        format_strs=format_strs
    )

  def run(self):
    if self.should_push_model:
      self._model_pool_apis.push_model(self.read_params(), None, self.model_key,
                                       learner_meta=self.read_opt_params())
    self.tstart = time.time()
    self.tfirststart = self.tstart
    self.total_samples = self._total_samples()
    train_losses, elapsed_time = [], 0
    for i in range(self._num_sgd_updates):
      # checkpoint stuff
      if i % self._checkpoint_interval == 0:
        if self._checkpoints_dir is not None:
          self._saver._save_model_checkpoint(self._checkpoints_dir,
                                             "checkpoint_%s" % i)
        while not self.data_pool.ready_for_val:
          time.sleep(5)
        if has_hvd:
          self.barrier()
        t = time.time()
        val_losses = self._validate()
        if self.rank == 0:
          logger.log("Validation Update: %d Val Loss: %s. "
              "Elapsed Time: %.2f" % (i, list(zip(self._names, val_losses)),
                                      time.time() - t))
          #self.print_logs(val_losses, self._num_val_batches)
        while not self.data_pool.ready_for_train:
          time.sleep(5)
        if has_hvd:
          self.barrier()
      # publish stuff
      if i % self.pub_interval == 0 and self.should_push_model:
        self._model_pool_apis.push_model(
          self.read_params(), None, self.model_key,
          learner_meta=self.read_opt_params()
        )
      loss = self._train_step()
      train_losses.append(loss)
      if len(train_losses) >= self._print_interval:
        if self.rank == 0:
          losses = [sum(ls) / len(ls) for ls in list(zip(*train_losses))]
          logger.logkvs({
            "n_update": i,
          })
          self.print_logs(losses, self._print_interval)
        train_losses = []

  def _validate(self):
    losses = []
    for _ in range(self._num_val_batches):
      loss = self._val_step()
      losses.append(loss)
    losses = list(zip(*losses))
    return [sum(ls) / len(ls) for ls in losses]

  def print_logs(self, losses, batchs):
    tnow = time.time()
    consuming_fps = int(
      self._batch_size * batchs / (tnow - self.tstart)
    )
    time_elapsed = tnow - self.tfirststart
    total_samples_now = self._total_samples()
    receiving_fps = ((sum(total_samples_now) - sum(self.total_samples))
                     / (tnow - self.tstart))
    self.total_samples = total_samples_now
    self.tstart = time.time()
    logger.logkvs({
      "all_consuming_fps": consuming_fps,
      "time_elapsed": time_elapsed,
      "train_samples": self.total_samples[0],
      "val_samples": self.total_samples[1],
      "receiving_fps": receiving_fps,
      })
    logger.logkvs({lossname: lossval for lossname, lossval
                  in zip(self._names, losses)})
    logger.dumpkvs()
