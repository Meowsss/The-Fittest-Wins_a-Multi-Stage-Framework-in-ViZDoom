import random
import numpy as np


class ReplayMem(object):
  """Replay Memory. data_queue is a list of unrolls. Each unroll has unroll_length samples,
     Each batch samples several rollouts, where each rollout has rollout_length consecutive samples."""

  def __init__(self, rm_size, unroll_length, batch_size, rollout_length, minimal_sample=None):
    self._rm_size = rm_size
    self._maxlen = int(np.ceil(rm_size / float(unroll_length)))
    if minimal_sample is not None:
      self._minimal_unroll = np.ceil(minimal_sample / float(unroll_length))
    else:
      self._minimal_unroll = self._maxlen
    self._unroll_length = unroll_length
    self._batch_size = batch_size
    self._rollout_length = rollout_length
    assert unroll_length % rollout_length == 0
    assert batch_size % rollout_length == 0
    self._rollout_per_unroll = unroll_length // rollout_length
    self._data_queue = []
    self._next_idx = 0
    self._ready = False  # do not sample before replay_memory is ready

  def __len__(self):
    return len(self._data_queue)

  def reset(self):
    self._data_queue = []
    self._next_idx = 0
    self._ready = False

  def ready_for_sample(self):
    if not self._ready:
      self._ready = len(self._data_queue) >= self._minimal_unroll
    return self._ready

  def append(self, data):
    idx = self._next_idx
    self._next_idx = (self._next_idx + 1) % self._maxlen
    if idx >= len(self._data_queue):
      self._data_queue.append(data)
    else:
      self._data_queue[idx] = data

  def sample_rollout(self):
    i = random.randint(0, len(self._data_queue) - 1)
    unroll = self._data_queue[i]  # backup the unroll in case of overwritten by new unroll
    j = random.randint(0, self._rollout_per_unroll - 1) * self._rollout_length
    for k in range(self._rollout_length):
      yield unroll[j + k]


class ImpSampReplayMem(ReplayMem):
  """Replay Memory with important sampling. """
  def __init__(self, rm_size, unroll_length, batch_size, rollout_length, minimal_sample=None):
    super(ImpSampReplayMem, self).__init__(rm_size, unroll_length, batch_size,
                                           rollout_length, minimal_sample)
    self._weights_queue = []
    self._unroll_weights = []

  def reset(self):
    super(ImpSampReplayMem, self).reset()
    self._weights_queue = []
    self._unroll_weights = []

  def append(self, data, weights):
    assert len(data) == len(weights)
    indx = self._next_idx
    self._next_idx = (self._next_idx + 1) % self._maxlen
    if indx >= len(self._data_queue):
      self._unroll_weights.append(sum(weights))
      self._weights_queue.append(weights)
      self._data_queue.append(data)
    else:
      self._unroll_weights[indx] = sum(weights)
      self._weights_queue[indx] = weights
      self._data_queue[indx] = data

  def sample_rollout(self):
    curr_data_size = len(self._data_queue)
    w = self._unroll_weights[:curr_data_size]
    p = np.array(w)
    i = np.random.choice(curr_data_size, p=p/np.sum(p))
    unroll = self._data_queue[i]  # backup the unroll in case of overwritten by new unroll
    w = np.array(self._weights_queue[i])
    w = np.reshape(w, (-1, self._rollout_length))
    p = np.sum(w, axis=1)
    j = np.random.choice(self._unroll_length//self._rollout_length, p=p/np.sum(p))
    l = j * self._rollout_length
    avg_w = p[j]/self._rollout_length
    for k in range(self._rollout_length):
      yield unroll[l + k], w[j, k]/avg_w
