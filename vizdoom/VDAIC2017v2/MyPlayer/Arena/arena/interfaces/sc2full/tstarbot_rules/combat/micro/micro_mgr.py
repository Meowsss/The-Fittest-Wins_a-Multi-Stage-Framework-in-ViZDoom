from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.lib.typeenums import UNIT_TYPEID

from .micro_base import MicroBase
from .roach_micro import RoachMgr
from .lurker_micro import LurkerMgr
from .mutalisk_micro import MutaliskMgr
from .ravager_micro import RavagerMgr
from .viper_micro import ViperMgr
from .corruptor_micro import CorruptorMgr
from .infestor_micro import InfestorMgr
from .queen_micro import QueenMgr
from .overseer_micro import OverseerMgr


class MicroMgr(MicroBase):
  """ A zvz Zerg combat manager """

  def __init__(self, dc):
    super(MicroMgr, self).__init__()
    self.roach_mgr = RoachMgr()
    self.lurker_mgr = LurkerMgr()
    self.mutalisk_mgr = MutaliskMgr()
    self.ravager_mgr = RavagerMgr()
    self.viper_mgr = ViperMgr()
    self.corruptor_mgr = CorruptorMgr()
    self.infestor_mgr = InfestorMgr()
    self.queen_mgr = QueenMgr()
    self.overseer_mgr = OverseerMgr()

    self.default_micro_version = 3
    self.init_config(dc)

  def init_config(self, dc):
    if hasattr(dc, 'config'):
      if hasattr(dc.config, 'default_micro_version'):
        self.default_micro_verion = int(dc.config.default_micro_version)

  def exe(self, dc, u, pos, mode):
    if u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_ROACH.value,
      UNIT_TYPEID.ZERG_ROACHBURROWED.value]:
      self.roach_mgr.update(dc)
      action = self.roach_mgr.act(u, pos, mode)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_LURKERMP.value,
      UNIT_TYPEID.ZERG_LURKERMPBURROWED.value]:
      self.lurker_mgr.update(dc)
      action = self.lurker_mgr.act(u, pos, mode)
    # elif u.int_attr.unit_type in [
    #   UNIT_TYPEID.ZERG_MUTALISK.value]:
    #   #self.mutalisk_mgr.update(dc)
    #   #action = self.mutalisk_mgr.act(u, pos, mode)
    #   self.update(dc)
    #   action = self.default_act(u, pos, mode)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_RAVAGER.value]:
      self.ravager_mgr.update(dc)
      action = self.ravager_mgr.act(u, pos, mode)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_VIPER.value]:
      self.viper_mgr.update(dc)
      action = self.viper_mgr.act(u, pos, mode)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_CORRUPTOR.value]:
      self.corruptor_mgr.update(dc)
      action = self.corruptor_mgr.act(u, pos, mode)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_INFESTOR.value]:
      self.infestor_mgr.update(dc)
      action = self.infestor_mgr.act(u, pos, mode)
    # elif u.int_attr.unit_type in [
    #   UNIT_TYPEID.ZERG_ULTRALISK.value]:
    #   self.update(dc)
    #   action = self.attack_pos(u, pos)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_QUEEN.value]:
      self.queen_mgr.update(dc)
      action = self.queen_mgr.act(u, pos, mode)
    elif u.int_attr.unit_type in [
      UNIT_TYPEID.ZERG_OVERSEER.value]:
      self.overseer_mgr.update(dc)
      action = self.overseer_mgr.act(u, pos, mode)
    else:
      self.update(dc)
      if pos is None:
        if u.int_attr.unit_type in [
          UNIT_TYPEID.ZERG_MUTALISK.value,
          UNIT_TYPEID.ZERG_HYDRALISK.value]:
          if len(self.enemy_combat_units) > 0:
            closest_enemy = self.find_closest_enemy(u, self.enemy_combat_units)
            pos = {'x': closest_enemy.float_attr.pos_x,
                   'y': closest_enemy.float_attr.pos_y}
          elif len(self.enemy_units) > 0:
            closest_enemy = self.find_closest_enemy(u, self.enemy_units)
            pos = {'x': closest_enemy.float_attr.pos_x,
                   'y': closest_enemy.float_attr.pos_y}
          else:
            pos = {'x': u.float_attr.pos_x,
                   'y': u.float_attr.pos_y}
        else:
          if len(self.enemy_ground_combat_units) > 0:
            closest_enemy = self.find_closest_enemy(u, self.enemy_ground_combat_units)
            pos = {'x': closest_enemy.float_attr.pos_x,
                   'y': closest_enemy.float_attr.pos_y}
          elif len(self.enemy_ground_units) > 0:
            closest_enemy = self.find_closest_enemy(u, self.enemy_ground_units)
            pos = {'x': closest_enemy.float_attr.pos_x,
                   'y': closest_enemy.float_attr.pos_y}
          else:
            pos = {'x': u.float_attr.pos_x,
                   'y': u.float_attr.pos_y}
      if u.int_attr.unit_type in [
        UNIT_TYPEID.ZERG_ULTRALISK.value,
        UNIT_TYPEID.ZERG_ZERGLING.value]:
        action = self.attack_pos(u, pos)
      elif u.int_attr.unit_type in [
        UNIT_TYPEID.ZERG_MUTALISK.value,
        UNIT_TYPEID.ZERG_HYDRALISK.value]:
        action = self.default_act(u, pos, mode)
      elif self.default_micro_version == 1:
        action = self.default_act(u, pos, mode)
      elif self.default_micro_version == 2:
        action = self.default_act_v2(u, pos, mode)
      elif self.default_micro_version == 3:
        action = self.attack_pos(u, pos)
      else:
        raise NotImplementedError
    return action

  def default_act(self, u, pos, mode):
    if len(self.enemy_combat_units) > 0:
      closest_enemy = self.find_closest_enemy(u, self.enemy_combat_units)
      if self.is_run_away(u, closest_enemy, self.self_combat_units):
        action = self.run_away_from_closest_enemy(u, closest_enemy)
      else:
        action = self.attack_pos(u, pos)
    else:
      action = self.attack_pos(u, pos)
    return action

  def default_act_v2(self, u, pos, mode):
    def POSX(u):
      return u.float_attr.pos_x

    def POSY(u):
      return u.float_attr.pos_y

    atk_range = self.get_atk_range(u.int_attr.unit_type)
    atk_type = self.get_atk_type(u.int_attr.unit_type)
    if not atk_range or not atk_type:
      return self.default_act(u, pos, mode)
    if len(self.enemy_combat_units) > 0:
      if self.ready_to_atk(u):
        weakest = self.find_weakest_nearby(u, self.enemy_combat_units,
                                           atk_range)
        if weakest:
          return self.attack_target(u, weakest)
        else:
          return self.attack_pos(u, pos)
      else:
        weakest = self.find_weakest_nearby(u, self.enemy_combat_units, 10)
        closest_enemy = self.find_closest_enemy(u, self.enemy_combat_units)
        if not weakest:
          return self.attack_pos(u, pos)
        enemy_range = self.get_atk_range(weakest.int_attr.unit_type)
        if self.is_run_away(u, closest_enemy, self.self_combat_units):
          return self.run_away_from_closest_enemy(u, closest_enemy)
        cur_dist = self.dist_between_units_with_radius(u, weakest)
        if enemy_range and atk_range >= enemy_range:
          if cur_dist < atk_range:
            return self.move_dir(u, (
            POSX(u) - POSX(weakest), POSY(u) - POSY(weakest)))
          else:
            return self.move_dir(u, (
            POSX(weakest) - POSX(u), POSY(weakest) - POSY(u)))
        else:
          return self.move_dir(u, (
          POSX(weakest) - POSX(u), POSY(weakest) - POSY(u)))
    else:
      action = self.attack_pos(u, pos)
    return action
