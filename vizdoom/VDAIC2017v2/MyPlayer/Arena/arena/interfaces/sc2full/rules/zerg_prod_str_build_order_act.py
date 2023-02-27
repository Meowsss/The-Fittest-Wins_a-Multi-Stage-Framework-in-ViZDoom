"""Production Strategy Manager"""
import random

from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import UPGRADE_ID
from pysc2.lib.typeenums import ABILITY_ID

from ....utils.constant import AllianceType as ALLY_TYPE
from ..tstarbot_rules.production_strategy.build_cmd import *
from ..tstarbot_rules.production_strategy.base_zerg_production_mgr import ZergBaseProductionMgr


def count_units(units, unit_type, ability_id=None):
  count = 0
  for u in units:
    if u.int_attr.alliance == ALLY_TYPE.SELF.value:
      if u.unit_type == unit_type.value:  # existing
        count += 1
      elif u.unit_type == UNIT_TYPE.ZERG_EGG.value:  # becoming
        if ability_id:
          if u.orders[0].ability_id == ability_id.value:
            count += 1
  return count


def count_units_types(units, unit_types, ability_ids=None):
  count = 0
  for u in units:
    if u.int_attr.alliance == ALLY_TYPE.SELF.value:
      if u.unit_type in [item.value for item in unit_types]:  # existing
        count += 1
      elif u.unit_type == UNIT_TYPE.ZERG_EGG.value:  # becoming
        if ability_ids:
          if u.orders[0].ability_id in [item.value for item in ability_ids]:
            count += 1
  return count


class BuildOrderStrategyLing(object):
  def get_cur_item(self, dc):
    units = dc.units
    obs = dc.obs
    n_drone = count_units(units, UNIT_TYPE.ZERG_DRONE, ABILITY_ID.TRAIN_DRONE)
    n_overlord = count_units(units, UNIT_TYPE.ZERG_OVERLORD,
                             ABILITY_ID.TRAIN_OVERLORD)
    n_spawningpool = count_units(units, UNIT_TYPE.ZERG_SPAWNINGPOOL,
                                 ABILITY_ID.BUILD_SPAWNINGPOOL)
    n_zergling = count_units(units, UNIT_TYPE.ZERG_ZERGLING,
                             ABILITY_ID.TRAIN_ZERGLING)
    n_spinecrawler = count_units(units, UNIT_TYPE.ZERG_SPINECRAWLER,
                                 ABILITY_ID.BUILD_SPINECRAWLER)
    n_base = count_units_types(
      units,
      [UNIT_TYPE.ZERG_HATCHERY, UNIT_TYPE.ZERG_HIVE, UNIT_TYPE.ZERG_LAIR],
      [ABILITY_ID.MORPH_LAIR, ABILITY_ID.MORPH_HIVE]
    )
    n_extractor = count_units(units, UNIT_TYPE.ZERG_EXTRACTOR,
                              ABILITY_ID.BUILD_EXTRACTOR)
    n_food_used = obs["player"][3]
    n_food_cap = obs["player"][4]

    if n_overlord < 2:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 15:
      return UNIT_TYPE.ZERG_DRONE
    elif n_spawningpool < 1:
      return UNIT_TYPE.ZERG_SPAWNINGPOOL
    elif n_drone < 18:
      return UNIT_TYPE.ZERG_DRONE
    elif n_overlord < 3:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_zergling < 6:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_base < 2:
      return UNIT_TYPE.ZERG_HATCHERY
    elif n_overlord < 4:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 21:
      return UNIT_TYPE.ZERG_DRONE
    elif n_extractor < 1:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_zergling < 12:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_drone < 22:
      return UNIT_TYPE.ZERG_DRONE
    elif n_base < 3:
      return UNIT_TYPE.ZERG_HATCHERY
    elif n_zergling < 24:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_drone < 30:
      return UNIT_TYPE.ZERG_DRONE
    elif n_zergling < 32:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_drone < 40:
      return UNIT_TYPE.ZERG_DRONE
    elif n_zergling < 50:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_zergling < 51:
      return UPGRADE_ID.ZERGLINGMOVEMENTSPEED

    # elif n_spinecrawler < 1:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 2:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 3:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_zergling < 18:
    #   return UNIT_TYPE.ZERG_ZERGLING
    # elif n_spinecrawler < 4:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 5:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 6:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_zergling < 22:
    #   return UNIT_TYPE.ZERG_ZERGLING
    # elif n_spinecrawler < 7:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 8:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_zergling < 30:
    #   return UNIT_TYPE.ZERG_ZERGLING

    if n_food_cap < 200:
      if n_food_cap - n_food_used < 4:
        return UNIT_TYPE.ZERG_OVERLORD

    return UNIT_TYPE.ZERG_ZERGLING


class BuildOrderStrategyHydralisk(object):
  def get_cur_item(self, dc):
    units = dc.units
    obs = dc.obs
    n_drone = count_units(units, UNIT_TYPE.ZERG_DRONE, ABILITY_ID.TRAIN_DRONE)
    n_overlord = count_units(units, UNIT_TYPE.ZERG_OVERLORD,
                             ABILITY_ID.TRAIN_OVERLORD)
    n_spawningpool = count_units(units, UNIT_TYPE.ZERG_SPAWNINGPOOL,
                                 ABILITY_ID.BUILD_SPAWNINGPOOL)
    n_hydraliskden = count_units(units, UNIT_TYPE.ZERG_HYDRALISKDEN,
                                 ABILITY_ID.BUILD_HYDRALISKDEN)
    n_zergling = count_units(units, UNIT_TYPE.ZERG_ZERGLING,
                             ABILITY_ID.TRAIN_ZERGLING)
    n_hydralisk = count_units(units, UNIT_TYPE.ZERG_HYDRALISK,
                              ABILITY_ID.TRAIN_HYDRALISK)
    n_spinecrawler = count_units(units, UNIT_TYPE.ZERG_SPINECRAWLER,
                                 ABILITY_ID.BUILD_SPINECRAWLER)
    n_base = count_units_types(
      units,
      [UNIT_TYPE.ZERG_HATCHERY, UNIT_TYPE.ZERG_HIVE, UNIT_TYPE.ZERG_LAIR],
      [ABILITY_ID.MORPH_LAIR, ABILITY_ID.MORPH_HIVE]
    )
    n_lair = count_units(units, UNIT_TYPE.ZERG_LAIR, ABILITY_ID.MORPH_LAIR)
    n_extractor = count_units(units, UNIT_TYPE.ZERG_EXTRACTOR,
                              ABILITY_ID.BUILD_EXTRACTOR)
    n_food_used = obs["player"][3]
    n_food_cap = obs["player"][4]

    if n_overlord < 2:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 15:
      return UNIT_TYPE.ZERG_DRONE
    elif n_spawningpool < 1:
      return UNIT_TYPE.ZERG_SPAWNINGPOOL
    elif n_drone < 18:
      return UNIT_TYPE.ZERG_DRONE
    elif n_overlord < 3:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_zergling < 6:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_base < 2:
      return UNIT_TYPE.ZERG_HATCHERY
    elif n_drone < 22:
      return UNIT_TYPE.ZERG_DRONE
    elif n_extractor < 1:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_spinecrawler < 1:
      return UNIT_TYPE.ZERG_SPINECRAWLER
    elif n_extractor < 2:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_lair < 1:
      return UNIT_TYPE.ZERG_LAIR
    elif n_hydraliskden < 1:
      return UNIT_TYPE.ZERG_HYDRALISKDEN
    elif n_hydralisk < 3:
      return UNIT_TYPE.ZERG_HYDRALISK
    elif n_extractor < 2:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_overlord < 4:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 25:
      return UNIT_TYPE.ZERG_DRONE
    elif n_hydralisk < 5:
      return UNIT_TYPE.ZERG_HYDRALISK
    elif n_overlord < 5:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 30:
      return UNIT_TYPE.ZERG_DRONE
    elif n_extractor < 3:
      return UNIT_TYPE.ZERG_EXTRACTOR

    if n_food_cap < 200:
      if n_food_cap - n_food_used < 4:
        return UNIT_TYPE.ZERG_OVERLORD

    return UNIT_TYPE.ZERG_HYDRALISK


class BuildOrderStrategyRoach(object):
  def get_cur_item(self, dc):
    units = dc.units
    obs = dc.obs
    n_drone = count_units(units, UNIT_TYPE.ZERG_DRONE, ABILITY_ID.TRAIN_DRONE)
    n_overlord = count_units(units, UNIT_TYPE.ZERG_OVERLORD,
                             ABILITY_ID.TRAIN_OVERLORD)
    n_spawningpool = count_units(units, UNIT_TYPE.ZERG_SPAWNINGPOOL,
                                 ABILITY_ID.BUILD_SPAWNINGPOOL)
    n_roachwarren = count_units(units, UNIT_TYPE.ZERG_ROACHWARREN,
                                ABILITY_ID.BUILD_ROACHWARREN)
    n_zergling = count_units(units, UNIT_TYPE.ZERG_ZERGLING,
                             ABILITY_ID.TRAIN_ZERGLING)
    n_spinecrawler = count_units(units, UNIT_TYPE.ZERG_SPINECRAWLER,
                                 ABILITY_ID.BUILD_SPINECRAWLER)
    n_roach = count_units(units, UNIT_TYPE.ZERG_ROACH, ABILITY_ID.TRAIN_ROACH)
    n_base = count_units_types(
      units,
      [UNIT_TYPE.ZERG_HATCHERY, UNIT_TYPE.ZERG_HIVE, UNIT_TYPE.ZERG_LAIR],
      [ABILITY_ID.MORPH_LAIR, ABILITY_ID.MORPH_HIVE]
    )
    n_lair = count_units(units, UNIT_TYPE.ZERG_LAIR, ABILITY_ID.MORPH_LAIR)
    n_extractor = count_units(units, UNIT_TYPE.ZERG_EXTRACTOR,
                              ABILITY_ID.BUILD_EXTRACTOR)
    n_food_used = obs["player"][3]
    n_food_cap = obs["player"][4]

    if n_overlord < 2:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 15:
      return UNIT_TYPE.ZERG_DRONE
    elif n_spawningpool < 1:
      return UNIT_TYPE.ZERG_SPAWNINGPOOL
    elif n_drone < 18:
      return UNIT_TYPE.ZERG_DRONE
    elif n_zergling < 6:
      return UNIT_TYPE.ZERG_ZERGLING
    elif n_overlord < 3:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_base < 2:
      return UNIT_TYPE.ZERG_HATCHERY
    elif n_drone < 22:
      return UNIT_TYPE.ZERG_DRONE
    elif n_extractor < 1:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_roachwarren < 1:
      return UNIT_TYPE.ZERG_ROACHWARREN
    elif n_extractor < 2:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_roach < 3:
      return UNIT_TYPE.ZERG_ROACH
    elif n_extractor < 2:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_overlord < 4:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 25:
      return UNIT_TYPE.ZERG_DRONE
    elif n_roach < 5:
      return UNIT_TYPE.ZERG_ROACH
    elif n_overlord < 5:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone< 30:
      return UNIT_TYPE.ZERG_DRONE
    elif n_extractor < 2:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_drone < 35:
      return UNIT_TYPE.ZERG_DRONE
    # elif n_spinecrawler < 1:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 2:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_roach < 6:
    #   return UNIT_TYPE.ZERG_ROACH
    # elif n_spinecrawler < 3:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 4:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_roach < 8:
    #   return UNIT_TYPE.ZERG_ROACH
    # elif n_spinecrawler < 5:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER
    # elif n_spinecrawler < 6:
    #   return UNIT_TYPE.ZERG_SPINECRAWLER

    if n_food_cap < 200:
      if n_food_cap - n_food_used < 4:
        return UNIT_TYPE.ZERG_OVERLORD

    return UNIT_TYPE.ZERG_ROACH


class BuildOrderStrategyMutualisk(object):
  def get_cur_item(self, dc):
    units = dc.units
    n_drone = count_units(units, UNIT_TYPE.ZERG_DRONE, ABILITY_ID.TRAIN_DRONE)
    n_overlord = count_units(units, UNIT_TYPE.ZERG_OVERLORD,
                             ABILITY_ID.TRAIN_OVERLORD)
    n_spawningpool = count_units(units, UNIT_TYPE.ZERG_SPAWNINGPOOL,
                                 ABILITY_ID.BUILD_SPAWNINGPOOL)
    n_spinecrawler = count_units(units, UNIT_TYPE.ZERG_SPINECRAWLER,
                                 ABILITY_ID.BUILD_SPINECRAWLER)
    n_zergling = count_units(units, UNIT_TYPE.ZERG_ZERGLING,
                             ABILITY_ID.TRAIN_ZERGLING)
    n_base = count_units_types(
      units,
      [UNIT_TYPE.ZERG_HATCHERY, UNIT_TYPE.ZERG_HIVE, UNIT_TYPE.ZERG_LAIR],
      [ABILITY_ID.MORPH_LAIR, ABILITY_ID.MORPH_HIVE]
    )
    n_lair = count_units(units, UNIT_TYPE.ZERG_LAIR, ABILITY_ID.MORPH_LAIR)
    n_extractor = count_units(units, UNIT_TYPE.ZERG_EXTRACTOR,
                              ABILITY_ID.BUILD_EXTRACTOR)
    n_spire = count_units(units, UNIT_TYPE.ZERG_SPIRE, ABILITY_ID.BUILD_SPIRE)
    n_food_used = dc.obs["player"][3]
    n_food_cap = dc.obs["player"][4]

    if n_drone < 14:
      return UNIT_TYPE.ZERG_DRONE
    elif n_spawningpool < 1:
      return UNIT_TYPE.ZERG_SPAWNINGPOOL
    elif n_drone < 14:
      return UNIT_TYPE.ZERG_DRONE
    elif n_base < 2:
      return UNIT_TYPE.ZERG_HATCHERY
    elif n_extractor < 1:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_spinecrawler < 1:
      return UNIT_TYPE.ZERG_SPINECRAWLER
    elif n_overlord < 2:
      return UNIT_TYPE.ZERG_OVERLORD
    elif n_drone < 22:
      return UNIT_TYPE.ZERG_DRONE
    elif n_spinecrawler < 2:
      return UNIT_TYPE.ZERG_SPINECRAWLER
    elif n_extractor < 2:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_lair < 1:
      return UNIT_TYPE.ZERG_LAIR
    elif n_extractor < 3:
      return UNIT_TYPE.ZERG_EXTRACTOR
    elif n_spire < 1:
      return UNIT_TYPE.ZERG_SPIRE
    elif n_extractor < 4:
      return UNIT_TYPE.ZERG_EXTRACTOR

    if n_food_cap < 200:
      if n_food_cap - n_food_used < 4:
        return UNIT_TYPE.ZERG_OVERLORD

    return UNIT_TYPE.ZERG_MUTALISK


class ZergProdStrBuildOrderActMgr(ZergBaseProductionMgr):
  """build order as action"""
  def __init__(self, dc, build_order_strategies, act_freq_game_loop):
    super(ZergProdStrBuildOrderActMgr, self).__init__(dc)
    self._act_freq_game_loop = act_freq_game_loop
    self._build_order_strategies = build_order_strategies
    self._cur_build_order_strategy = None
    self._cur_step = 0

    self.verbose = 0
    self.action_space = spaces.Discrete(len(self._build_order_strategies))

  def reset_by_obs(self, obs):
    self.onStart = True
    self.build_order.clear_all()
    self._cur_build_order_strategy = None
    self._cur_step = 0
    self.obs = obs.observation
    self.cut_in_item = []
    self.born_pos = None

  def update_by_action(self, dc, am, action):
    # initialize base position
    if self.onStart:
      bases = dc.dd.base_pool.bases
      self.born_pos = [
        [bases[tag].unit.float_attr.pos_x, bases[tag].unit.float_attr.pos_y] for tag in bases
      ][0]
      self.onStart = False

    # update observation
    self.obs = dc.sd.obs

    # update current build_order_strategy by given action!
    if self._cur_step == 0 or self.obs['game_loop'] % self._act_freq_game_loop == 0:
      self._cur_build_order_strategy = self._build_order_strategies[action]

    # update
    if self.build_order.is_empty():
      build_item = self._cur_build_order_strategy.get_cur_item(dc=dc)
      self.build_order.queue_as_highest(build_item)

    # auto supply
    self.check_supply()

    # push the concrete command
    if not self.build_order.is_empty():
      while self.detect_dead_lock(dc):
        pass
      current_item = self.build_order.current_item()
      if current_item and self.can_build(current_item, dc):
        # print(current_item.unit_id)
        if current_item.isUnit:
          if self.set_build_base(current_item, dc):
            self.build_order.remove_current_item()
            if current_item.unit_id in self.cut_in_item:
              self.cut_in_item.remove(current_item.unit_id)
            if self.verbose > 0:
              print('Produce: {}'.format(current_item.unit_id))
        else:  # Upgrade
          if self.upgrade(current_item, dc):
            self.build_order.remove_current_item()
            if current_item.unit_id in self.cut_in_item:
              self.cut_in_item.remove(current_item.unit_id)
            if self.verbose > 0:
              print('Upgrade: {}'.format(current_item.unit_id))

    # keep_order:
    self.build_order.clear_all()
    self.cut_in_item = []

    # auto resource
    if self.should_gas_first(dc):
      dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=True))
    if self.should_mineral_first(dc):
      dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=False))

    # auto larva
    self.spawn_larva(dc)

    # update step
    self._cur_step += 1

  def detect_dead_lock(self, data_context):
    current_item = self.build_order.current_item()
    builder = None
    for unit_type in current_item.whatBuilds:
      if (self.has_unit(unit_type) or self.unit_in_progress(unit_type)
          or unit_type == UNIT_TYPE.ZERG_LARVA.value):
        builder = unit_type
        break
    if builder is None and len(current_item.whatBuilds) > 0:
      builder_id = [unit_id for unit_id in UNIT_TYPE
                    if unit_id.value == current_item.whatBuilds[0]]
      self.build_order.queue_as_highest(builder_id[0])
      if self.verbose > 0:
        print('Cut in: {}'.format(builder_id[0]))
      return True
    required_unit = None
    for unit_type in current_item.requiredUnits:
      if self.has_unit(unit_type) or self.unit_in_progress(unit_type):
        required_unit = unit_type
        break
    if required_unit is None and len(current_item.requiredUnits) > 0:
      required_id = [unit_id for unit_id in UNIT_TYPE
                     if unit_id.value == current_item.requiredUnits[0]]
      self.build_order.queue_as_highest(required_id[0])
      if self.verbose > 0:
        print('Cut in: {}'.format(required_id[0]))
      return True
    required_upgrade = None
    for upgrade_type in current_item.requiredUpgrades:
      if (not self.has_upgrade(data_context, [upgrade_type])
          and not self.upgrade_in_progress(upgrade_type)):
        required_upgrade = upgrade_type
        break
    if required_upgrade is not None:
      required_id = [up_id for up_id in UPGRADE_ID
                     if up_id.value == required_upgrade]
      self.build_order.queue_as_highest(required_id[0])
      if self.verbose > 0:
        print('Cut in: {}'.format(required_id[0]))
      return True
    if current_item.gasCost > 0:
      n_g = len([u for u in self.obs['units']
                 if u.unit_type == UNIT_TYPE.ZERG_EXTRACTOR.value
                 and u.int_attr.vespene_contents > 100
                 and u.int_attr.alliance == 1])
      if n_g == 0:
        self.build_order.queue_as_highest(self.gas_unit())
        if self.verbose > 0:
          print('Cut in: {}'.format(self.gas_unit()))
        return True
    return False

  def can_build(self, build_item, dc):  # check resource requirement
    if not (self.has_building_built(build_item.whatBuilds)
            and self.has_building_built(build_item.requiredUnits)
            and self.has_upgrade(dc, build_item.requiredUpgrades)
            and not self.expand_waiting_resource(dc)):
      return False
    play_info = self.obs["player"]
    if (build_item.supplyCost > 0
        and play_info[3] + build_item.supplyCost > play_info[4]):
      return False
    if build_item.unit_id == UNIT_TYPE.ZERG_HATCHERY:
      return self.obs['player'][1] >= build_item.mineralCost - 100
    return (self.obs['player'][1] >= build_item.mineralCost
            and self.obs['player'][2] >= build_item.gasCost)
