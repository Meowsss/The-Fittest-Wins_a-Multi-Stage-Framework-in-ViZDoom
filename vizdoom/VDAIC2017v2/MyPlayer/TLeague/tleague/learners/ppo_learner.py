from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time

if sys.version_info.major > 2:
  xrange = range

import joblib
import numpy as np
import tensorflow as tf
from gym import spaces

from tleague.learners.base_learner import BaseLearner
from tleague.learners.data_server import DataServer
from tleague.utils import logger
from tleague.utils.data_structure import PPOData


def as_func(obj):
  if isinstance(obj, float):
    return lambda x: obj
  else:
    assert callable(obj)
    return obj


def average_tf(grads):
  avg_grads = []
  for g in zip(*grads):
    if g[0] is not None:
      grad = tf.stack(g)
      grad = tf.reduce_mean(grad, 0)
    else:
      grad = None
    avg_grads.append(grad)
  return avg_grads


class PPOLearner(BaseLearner):

  def __init__(self, league_mgr_addr, model_pool_addrs, learner_ports,
               rm_size, batch_size, ob_space, ac_space, policy,
               policy_config={}, ent_coef=1e-2, unroll_length=32,
               n_v=1, vf_coef=0.5, max_grad_norm=0.5,
               pub_interval=500, log_interval=100, save_interval=0,
               gpu_num=0, total_timesteps=5e7, burn_in_timesteps=0,
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

    self.sess = tf.Session()  # config=config)
    if self.rnn:
      policy_config['rollout_len'] = rollout_length

    ## Prepare dataset
    ds = PPOData(ob_space, ac_space, n_v, use_lstm=self.rnn, hs_len=self.hs_len,
                 distillation=self.distillation, version='v1')
    self._data_server = DataServer(self._pull_data, rm_size,
                                   unroll_length, batch_size, ds,
                                   gpu_id_list=range(gpu_num),
                                   batch_worker_num=batch_worker_num,
                                   pull_worker_num=pull_worker_num,
                                   rollout_length=rollout_length)

    ## Build policy
    self.num_towers = max(gpu_num, 1)
    grads_towers = []
    grads_vf_towers = []
    losses_towers = []

    with tf.device('/cpu:0'):
      model = policy(ob_space=ob_space, ac_space=ac_space, nbatch=batch_size,
                     n_v=n_v, input_data=None, reuse=tf.AUTO_REUSE,
                     scope_name='model', **policy_config)
    self.params = tf.trainable_variables(scope='model')
    self.params_vf = tf.trainable_variables(scope='model/vf')
    self.param_norm = tf.global_norm(self.params)

    for tower_id in range(self.num_towers):
      if gpu_num == 0:
        device = '/cpu:0'
      else:
        device = '/gpu:%d' % tower_id
      with tf.device(device):
        input_data = self._data_server.input_datas[tower_id]
        model = policy(ob_space=ob_space, ac_space=ac_space, nbatch=batch_size,
                       n_v=n_v, input_data=input_data, reuse=tf.AUTO_REUSE,
                       scope_name='model', **policy_config)
        loss, vf_loss, losses = self.build_loss(model, input_data)
        grads = tf.gradients(loss, self.params)
        grads_vf = tf.gradients(vf_loss, self.params_vf)
        grads_towers.append(grads)
        grads_vf_towers.append(grads_vf)
        losses_towers.append(losses)

    # average grads and clip
    with tf.device('/cpu:0'):
      self.losses = average_tf(losses_towers)
      avg_grads = average_tf(grads_towers)
      avg_grads_vf = average_tf(grads_vf_towers)
      if max_grad_norm is not None:
        avg_grads, self.grad_norm = tf.clip_by_global_norm(avg_grads, max_grad_norm)
        avg_grads = list(zip(avg_grads, self.params))
        avg_grads_vf, self.grad_vf_norm = tf.clip_by_global_norm(avg_grads_vf, max_grad_norm)
        avg_grads_vf = list(zip(avg_grads_vf, self.params_vf))
      else:
        self.grad_norm = tf.global_norm(avg_grads)
        avg_grads = list(zip(avg_grads, self.params))
        self.grad_vf_norm = tf.global_norm(avg_grads_vf)
        avg_grads_vf = list(zip(avg_grads_vf, self.params_vf))

    self.trainer = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
    self._train_batch = self.trainer.apply_gradients(avg_grads)
    self.burn_in_trainer = tf.train.AdamOptimizer(learning_rate=self.LR, epsilon=1e-5)
    self._burn_in = self.burn_in_trainer.apply_gradients(avg_grads_vf)

    self._build_ops()
    tf.global_variables_initializer().run(session=self.sess)
    self.sess.graph.finalize()
    logger.configure(dir='training_log/' + self._learner_id,
                     format_strs=['stdout', 'log', 'tensorboard', 'csv'])

  def _init_const(self, total_timesteps, burn_in_timesteps, batch_size,
                  unroll_length, ob_space, ac_space, rwd_shape,
                  merge_pi, n_v, adv_normalize, ent_coef, vf_coef,
                  pub_interval, log_interval, save_interval,
                  policy, policy_config, distill_coef):
    self.total_timesteps = total_timesteps
    self.burn_in_timesteps = burn_in_timesteps
    self._train_batch = []
    self._burn_in = []
    self.batch_size = batch_size
    self.unroll_length = unroll_length
    self.ob_space = ob_space
    self.ac_space = ac_space
    self.rwd_shape = rwd_shape
    self.merge_pi = merge_pi and isinstance(self.ac_space, spaces.Tuple)
    self.n_v = n_v
    self.adv_normalize = adv_normalize
    self.ent_coef = ent_coef
    self.vf_coef = vf_coef
    self.pub_interval = pub_interval
    self.log_interval = log_interval
    self.save_interval = save_interval
    self.policy = policy
    self.rnn = False if 'use_lstm' not in policy_config else policy_config['use_lstm']
    self.hs_len = None
    self.distillation = (distill_coef != 0)
    self.distill_coef = distill_coef
    if self.rnn:
      self.hs_len = (policy_config['hs_len'] if ('hs_len' in policy_config)
                     else 2 * policy_config['nlstm'] if ('nlstm' in policy_config) else 128)
    if isinstance(self.ac_space, spaces.Tuple) and not self.merge_pi:
      shape = [len(self.ac_space.spaces), n_v]
    else:
      shape = [1, n_v]
    self.rwd_weights = tf.placeholder(tf.float32, shape)
    self.LR = tf.placeholder(tf.float32, [])
    self.CLIPRANGE = tf.placeholder(tf.float32, [])

  def _build_ops(self):
    ## other useful operators
    self.new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in self.params]
    self.param_assign_ops = [p.assign(new_p) for p, new_p in zip(self.params, self.new_params)]
    self.opt_params = self.trainer.variables()
    self.new_opt_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in self.opt_params]
    self.opt_param_assign_ops = [p.assign(new_p) for p, new_p in zip(self.opt_params, self.new_opt_params)]
    self.reset_optimizer_op = tf.variables_initializer(
      self.trainer.variables() + self.burn_in_trainer.variables())

    self.loss_names = [
      'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac',
      'mean_return', 'explained_var', 'distill_loss', 'grad_norm', 'param_norm'
    ]

    def train_batch(lr, cliprange, weights=None):
      if weights is None:
        assert not self.rwd_shape
        td_map = {self.LR: lr, self.CLIPRANGE: cliprange}
      else:
        td_map = {self.LR: lr, self.CLIPRANGE: cliprange,
                  self.rwd_weights: weights}
      return self.sess.run(
        self.losses + [self.grad_norm, self.param_norm, self._train_batch],
        td_map
      )[0:len(self.loss_names)]

    def burn_in(lr, cliprange, weights=None):
      if weights is None:
        assert not self.rwd_shape
        td_map = {self.LR: lr, self.CLIPRANGE: cliprange}
      else:
        td_map = {self.LR: lr, self.CLIPRANGE: cliprange,
                  self.rwd_weights: weights}
      return self.sess.run(
        self.losses + [self.grad_vf_norm, self.param_norm, self._burn_in],
        td_map
      )[0:len(self.loss_names)]

    def save(save_path):
      ps = self.sess.run(self.params)
      joblib.dump(ps, save_path)

    def load_model(loaded_params):
      self.sess.run(self.param_assign_ops,
                    feed_dict={p: v for p, v in zip(self.new_params, loaded_params)})

    def restore_optimizer(loaded_opt_params):
      self.sess.run(self.opt_param_assign_ops,
                    feed_dict={p: v for p, v in zip(self.new_opt_params, loaded_opt_params)})

    def load(load_path):
      loaded_params = joblib.load(load_path)
      load_model(loaded_params)

    def read_params():
      return self.sess.run(self.params)

    def read_opt_params():
      return self.sess.run(self.opt_params)

    def reset():
      self.sess.run(self.reset_optimizer_op)

    self.train_batch = train_batch
    self.burn_in = burn_in
    self.save = save
    self.load_model = load_model
    self.restore_optimizer = restore_optimizer
    self.load = load
    self.read_params = read_params
    self.read_opt_params = read_opt_params
    self.reset = reset

  def _do_rwd_shape(self, r):
    r = tf.matmul(r, self.rwd_weights, transpose_b=True)
    return tf.squeeze(r)

  def build_loss(self, model, input_data):
    ADV = input_data.R - input_data.V
    mean_return, var_return = tf.nn.moments(input_data.R, axes=[0], keep_dims=True)
    if self.rwd_shape:
      # logger.log('ADV: ', tf.shape(ADV))
      ADV = self._do_rwd_shape(ADV)
      mean_return = self._do_rwd_shape(mean_return)
      # logger.log('ADV rwd_shape: ', tf.shape(ADV))
    mean_return = tf.reduce_mean(mean_return)
    neglogpac = model.pd.neglogp(input_data.A)
    entropy = tf.reduce_mean(model.pd.entropy(), axis=0)  # reduce mean at the batch dimension
    if self.merge_pi:
      ratio = tf.exp(tf.reduce_sum(input_data.neglogp - neglogpac, axis=-1))
      # logger.log('ratio reduced: ', str(tf.shape(ratio)))
    else:
      ratio = tf.exp(input_data.neglogp - neglogpac)
    # logger.log('ratio: ', str(tf.shape(ratio)))

    # normalize ADV
    ADV = ADV - tf.reduce_mean(ADV, axis=0)
    if self.adv_normalize:
      ADV = ADV / tf.sqrt(tf.reduce_mean(tf.square(ADV), axis=0) + 1e-8)
      # logger.log('ADV adv_normalize: ', tf.shape(ADV))

    vpred = model.vf
    vpredclipped = input_data.V + tf.clip_by_value(model.vf - input_data.V,
                                               - self.CLIPRANGE, self.CLIPRANGE)
    vf_losses1 = tf.square(vpred - input_data.R)
    vf_losses2 = tf.square(vpredclipped - input_data.R)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    pg_losses1 = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - self.CLIPRANGE,
                                         1.0 + self.CLIPRANGE)
    # logger.log('pg_losses1: ', str(tf.shape(pg_losses1)))
    # logger.log('pg_losses2: ', str(tf.shape(pg_losses2)))
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
    distill_loss = tf.constant(0, dtype=tf.float32)

    if self.distillation:
      distill_loss = tf.reduce_mean(model.kl(input_data.logits), axis=0)  # reduce mean at the batch dimension
    if isinstance(self.ent_coef, list):
      entropy_list = tf.unstack(entropy, axis=0)
      assert len(entropy_list) == len(self.ent_coef), 'Lengths of ent and ent_coef mismatch.'
      print('ent_coef: {}'.format(self.ent_coef))
      entropy = tf.add_n([e*ec for e, ec in zip(entropy_list, self.ent_coef)])
    else:
      entropy = tf.reduce_sum(entropy) * self.ent_coef
    if isinstance(self.distill_coef, list):
      distill_losses = tf.unstack(distill_loss, axis=0)
      assert len(distill_losses) == len(self.distill_coef), 'Lengths of distill and distill_coef mismatch.'
      print('distill_coef: {}'.format(self.distill_coef))
      distill_loss = tf.add_n([d*dc for d, dc in zip(distill_losses, self.distill_coef)])
    else:
      distill_loss = tf.reduce_sum(distill_loss) * self.distill_coef

    loss = (pg_loss - entropy + vf_loss * self.vf_coef + distill_loss)

    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - input_data.neglogp))
    mean_res, var_res = tf.nn.moments(vpred - input_data.R, axes=[0], keep_dims=True)
    explained_var = tf.reduce_mean(1 - var_res / var_return)
    clipfrac = tf.reduce_mean(
      tf.to_float(tf.greater((ratio - 1.0) * tf.sign(ADV), self.CLIPRANGE)))
    return loss, vf_loss, [pg_loss, vf_loss, entropy / np.sum(self.ent_coef), approxkl, clipfrac,
                           mean_return, explained_var, distill_loss / np.sum(self.distill_coef)]

  def _train(self, **kwargs):
    self._notify_task_begin(self.task)
    self._data_server._update_model_id(self.model_key)
    # Use different model, clear the replay memory
    if self.last_model_key is None or self.last_model_key != self.task.parent_model_key:
      self._data_server.reset()

    nbatch = self.batch_size * self.num_towers
    self.should_push_model = True
    self._run_train_loop(nbatch)

  def _run_train_loop(self, nbatch):
    lr = as_func(self.task.hyperparam.learning_rate)
    cliprange = as_func(self.task.hyperparam.cliprange)
    weights = None
    if self.rwd_shape:
      assert hasattr(self.task.hyperparam, 'reward_weights')
      weights = np.array(self.task.hyperparam.reward_weights, dtype=np.float32)
      if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 0)
    self.total_timesteps = getattr(self.task.hyperparam, 'total_timesteps', self.total_timesteps)
    self.burn_in_timesteps = getattr(self.task.hyperparam, 'burn_in_timesteps', self.burn_in_timesteps)
    nupdates_burn_in = int(self.burn_in_timesteps // nbatch)
    nupdates = nupdates_burn_in + int(self.total_timesteps // nbatch)
    mblossvals = []
    tfirststart = time.time()
    tstart = time.time()
    total_samples = self._data_server.unroll_num * self.unroll_length
    logger.log('Start Training')
    for update in xrange(1, nupdates + 1):
      frac = 1.0 - (update - 1.0) / nupdates
      lrnow = lr(frac)
      cliprangenow = cliprange(frac)
      if update <= nupdates_burn_in:
        mblossvals.append(self.burn_in(lrnow, cliprangenow, weights))
      else:
        mblossvals.append(self.train_batch(lrnow, cliprangenow, weights))
      # publish models
      if update % self.pub_interval == 0 and self.should_push_model:
        self._model_pool_apis.push_model(self.read_params(), self.task.hyperparam,
                                         self.model_key, learner_meta=self.read_opt_params())
      # logging stuff
      if update % self.log_interval == 0 or update == 1:
        lossvals = np.mean(mblossvals, axis=0)
        mblossvals = []
        tnow = time.time()
        consuming_fps = int(
          nbatch * min(update, self.log_interval) / (tnow - tstart)
        )
        time_elapsed = tnow - tfirststart
        total_samples_now = self._data_server.unroll_num * self.unroll_length
        receiving_fps = (total_samples_now - total_samples) / (tnow - tstart)
        total_samples = total_samples_now
        tstart = time.time()
        # 'scope_name/var' style for grouping Tab in Tensorboard webpage
        # lp is short for Learning Period
        scope = 'lp{}/'.format(self._lrn_period_count)
        logger.logkvs({
          scope + "lrn_period_count": self._lrn_period_count,
          scope + "burn_in_value": update <= nupdates_burn_in,
          scope + "nupdates": update,
          scope + "total_timesteps": update * nbatch,
          scope + "all_consuming_fps": consuming_fps,
          scope + 'time_elapsed': time_elapsed,
          scope + "total_samples": total_samples,
          scope + "receiving_fps": receiving_fps,
          scope + "aband_samples": (self._data_server.aband_unroll_num *
                                    self.unroll_length)
          })
        logger.logkvs({scope + lossname: lossval for lossname, lossval
                       in zip(self.loss_names, lossvals)})
        logger.dumpkvs()
      if self.save_interval and (
          update % self.save_interval == 0 or update == 1) and logger.get_dir():
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i' % update)
        logger.log('Saving log to', savepath)
        self.save(savepath)
    if self.should_push_model:
      self._model_pool_apis.push_model(self.read_params(), self.task.hyperparam,
                                       self.model_key, learner_meta=self.read_opt_params())
    return

  def _init_task(self):
    task = self.task
    logger.log('Period: {},'.format(self._lrn_period_count),
               'Task: {}'.format(str(task)))
    logger.log('Continue training from model: {}. New model id: {}.'.format(
      str(task.parent_model_key), str(self.model_key)))

    hyperparam = task.hyperparam
    if task.parent_model_key is None:
      logger.log(
        'Parent model is None, '
        'pushing new model {} params to ModelPool'.format(self.model_key)
      )
      self._model_pool_apis.push_model(self.read_params(), hyperparam,
                                       self.model_key, learner_meta=self.read_opt_params())
    elif self.model_key != self.last_model_key:
      logger.log(
        'Parent model {} exists, pulling its params from ModelPool '
        'as new model {}'.format(task.parent_model_key, self.model_key)
      )
      model_obj = self._model_pool_apis.pull_model(task.parent_model_key)
      self._model_pool_apis.push_model(model_obj.model, hyperparam,
                                       self.model_key, learner_meta=self.read_opt_params())
      self.load_model(model_obj.model)
      learner_meta = self._model_pool_apis.pull_learner_meta(task.parent_model_key)
      if learner_meta is not None:
        logger.log(
          'Restore optimizer from model {}'.format(task.parent_model_key)
        )
        self.restore_optimizer(learner_meta)
      else:
        self.reset()
    else:
      logger.log('Continue training model {}.'.format(self.model_key))
