"""data context"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.data_raw import get_data_raw
from pysc2.lib import TechTree

from .queue.build_command_queue import BuildCommandQueue
from .queue.build_command_queue import BuildCommandQueueV2
from .queue.combat_command_queue import CombatCommandQueue
from .queue.scout_command_queue import ScoutCommandQueue
from .pool.base_pool import BasePool
from .pool.building_pool import BuildingPool
from .pool.worker_pool import WorkerPool
from .pool.combat_pool import CombatUnitPool
from .pool.enemy_pool import EnemyPool
from .pool.scout_pool import ScoutPool
from .pool.opponent_pool import OppoPool


class StaticData(object):
  def __init__(self, config, game_version):
    self._obs = None
    self._timestep = None
    self._data_raw = get_data_raw(game_version)

    self.game_version = game_version
    self.TT = TechTree()
    self.TT.update_version(self.game_version)

  def update(self, timestep):
    self._obs = timestep.observation
    self._timestep = timestep

  @property
  def obs(self):
    return self._obs

  @property
  def timestep(self):
    return self._timestep

  @property
  def data_raw(self):
    return self._data_raw


class DynamicData(object):
  def __init__(self, config, game_version):
    self.game_version = game_version
    self.build_command_queue = BuildCommandQueueV2()
    self.combat_command_queue = CombatCommandQueue()
    self.scout_command_queue = ScoutCommandQueue()

    self.building_pool = BuildingPool()
    self.worker_pool = WorkerPool()
    self.combat_pool = CombatUnitPool()
    self.base_pool = BasePool(self)
    self.enemy_pool = EnemyPool(self)
    self.scout_pool = ScoutPool(self)
    self.oppo_pool = OppoPool()

  def update(self, timestep):
    # update command queues

    # update pools
    self.building_pool.update(timestep)
    self.worker_pool.update(timestep)
    self.combat_pool.update(timestep)
    self.base_pool.update(timestep)
    self.enemy_pool.update(timestep)
    self.scout_pool.update(timestep)

    # update statistic

  def reset(self):
    self.base_pool.reset()
    self.scout_pool.reset()
    self.oppo_pool.reset()
    self.enemy_pool.reset()


class DataContext:
  def __init__(self, config):
    self.config = config
    self.game_version = '3.16.1'
    if hasattr(config, 'game_version'):
      self.game_version = config.game_version
    self._dynamic = DynamicData(config, self.game_version)
    self._static = StaticData(config, self.game_version)

  def update(self, timestep):
    # self._obs = timestep.observation
    self._dynamic.update(timestep)
    self._static.update(timestep)

  def reset(self):
    # print('***DataContext reset***')
    self._dynamic.reset()

  @property
  def dd(self):
    return self._dynamic

  @property
  def sd(self):
    return self._static

  @property
  def obs(self):
    return self._static.obs

  @property
  def units(self):
    return self._static.obs['units']
