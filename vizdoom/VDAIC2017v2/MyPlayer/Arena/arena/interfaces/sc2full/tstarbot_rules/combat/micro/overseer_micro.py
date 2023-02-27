from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib.typeenums import UNIT_TYPEID, ABILITY_ID
from s2clientprotocol import sc2api_pb2 as sc_pb
import numpy as np

from .micro_base import MicroBase
from ...data.pool.macro_def import COMBAT_FLYING_UNITS


class OverseerMgr(MicroBase):
  """ A zvz Zerg combat manager """

  def __init__(self):
    super(OverseerMgr, self).__init__()

  def find_densest_enemy_pos(self, u):
    enemy_ground_units = [e for e in self.enemy_combat_units
                          if e.int_attr.unit_type not in COMBAT_FLYING_UNITS and
                          e.int_attr.unit_type not in [
                            UNIT_TYPEID.ZERG_SPINECRAWLER.value,
                            UNIT_TYPEID.ZERG_SPORECRAWLER.value]]
    if len(enemy_ground_units) == 0:
      return {'x': u.float_attr.pos_x,
              'y': u.float_attr.pos_y}
    target_density = list()
    for e in enemy_ground_units:
      target_density.append(len(
        self.find_units_wihtin_range(e, enemy_ground_units, r=20)))
    target_id = np.argmax(target_density)
    target = enemy_ground_units[target_id]
    target_pos = {'x': target.float_attr.pos_x,
                  'y': target.float_attr.pos_y}
    return target_pos

  def act(self, u, pos, mode):
    if pos is None:
      if len(self.enemy_combat_units) > 0:
        pos = self.find_densest_enemy_pos(u)
      else:
        pos = {'x': u.float_attr.pos_x,
               'y': u.float_attr.pos_y}
    front_units = []
    if len(self.self_combat_units) > 0:
      front_units = self.find_nearest_n_units_to_pos(
        pos, self.self_combat_units, min(5, len(self.self_combat_units)))
    if len(front_units) > 0:
      move_pos = self.get_center_of_units(self.self_combat_units)
      action = self.move_pos(u, move_pos)
    else:
      action = self.hold_fire(u)
    return action
