"""Combat Manager"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..data.pool.macro_def import UNIT_TYPEID
from ..data.queue.combat_command_queue import CombatCmdType
from ..combat.micro.micro_mgr import MicroMgr
from ..combat.micro.lurker_micro import LurkerMgr
from ..util import geom


class BaseCombatMgr(object):
  """ Basic Combat Manager

  Common Utilites for combat are implemented here. """

  def __init__(self, dc):
    pass

  def reset(self):
    pass

  def update(self, dc, am):
    pass


class ZergCombatMgr(BaseCombatMgr):
  """ A zvz Zerg combat manager """

  def __init__(self, dc):
    super(ZergCombatMgr, self).__init__(dc)
    self.dc = dc
    self.micro_mgr = MicroMgr(dc)

  def reset(self):
    self.micro_mgr = MicroMgr(self.dc)
    self.dc = None

  def update(self, dc, am):
    super(ZergCombatMgr, self).update(dc, am)
    self.dc = dc

    actions = list()
    while True:
      cmd = dc.dd.combat_command_queue.pull()
      if not cmd:
        break
      else:
        actions.extend(self.exe_cmd(cmd.squad, cmd.position, cmd.type))
    am.push_actions(actions)

  def reduce_redundant_apm(self, u, action):
    if not hasattr(u, 'orders'):
      return False

    action_cmd = action.action_raw.unit_command

    if len(u.orders) == 0:
      if hasattr(action_cmd, 'target_world_space_pos') and \
              (u.float_attr.pos_x - action_cmd.target_world_space_pos.x) ** 2 + \
              (u.float_attr.pos_y - action_cmd.target_world_space_pos.y) ** 2 < 25:
        return True
      else:
        return False

    if u.orders[0].ability_id == action_cmd.ability_id and \
            hasattr(action_cmd, 'target_world_space_pos') and \
            u.orders[0].target_pos_x == action_cmd.target_world_space_pos.x and \
            u.orders[0].target_pos_y == action_cmd.target_world_space_pos.y:
      return True
    if u.orders[0].ability_id == action_cmd.ability_id and \
            u.orders[0].target_tag != 0 and \
            hasattr(action_cmd, 'target_tag') and \
            u.orders[0].target_tag == action_cmd.target_tag:
      return True
    return False

  def exe_cmd(self, squad, pos, mode):
    actions = []
    if mode == CombatCmdType.ATTACK:
      actions = self.exe_attack(squad, pos)
    elif mode == CombatCmdType.MOVE:
      actions = self.exe_move(squad, pos)
    elif mode == CombatCmdType.DEFEND:
      actions = self.exe_defend(squad, pos)
    elif mode == CombatCmdType.RALLY:
      actions = self.exe_rally(squad, pos)
    elif mode == CombatCmdType.ROCK:
      actions = self.exe_rock(squad, pos)
    return actions

  def exe_attack(self, squad, pos):
    actions = list()
    squad_units = []
    for combat_unit in squad.units:
      squad_units.append(combat_unit.unit)
    for u in squad_units:
      action = self.exe_micro(u, pos, mode=CombatCmdType.ATTACK)
      if not self.reduce_redundant_apm(u, action):
        actions.append(action)
    return actions

  def exe_defend(self, squad, pos):
    actions = list()
    squad_units = []
    for combat_unit in squad.units:
      squad_units.append(combat_unit.unit)
    for u in squad_units:
      action = self.exe_micro(u, pos, mode=CombatCmdType.DEFEND)
      if not self.reduce_redundant_apm(u, action):
        actions.append(action)
    return actions

  def exe_move(self, squad, pos):
    actions = []
    for u in squad.units:
      u = u.unit
      if u.int_attr.unit_type == UNIT_TYPEID.ZERG_LURKERMPBURROWED.value:
        actions.append(LurkerMgr().burrow_up(u))
      else:
        action = self.micro_mgr.move_pos(u, pos)
        if not self.reduce_redundant_apm(u, action):
          actions.append(action)
    return actions

  def exe_rally(self, squad, pos):
    actions = []
    for u in squad.units:
      action = self.micro_mgr.attack_pos(u.unit, pos)
      if not self.reduce_redundant_apm(u, action):
        actions.append(action)
    return actions

  def exe_rock(self, squad, pos):
    actions = []
    rocks = [u for u in self.dc.sd.obs['units']
             if u.int_attr.unit_type ==
             UNIT_TYPEID.NEUTRAL_DESTRUCTIBLEROCKEX1DIAGONALHUGEBLUR.value]
    target_rock = None
    for r in rocks:
      d = geom.dist_to_pos(r, (pos['x'], pos['y']))
      if d < 0.1:
        target_rock = r
        break
    for u in squad.units:
      action = self.micro_mgr.attack_target(u, target_rock)
      if not self.reduce_redundant_apm(u, action):
        actions.append(action)
    return actions

  def exe_micro(self, u, pos, mode):
    action = self.micro_mgr.exe(self.dc, u, pos, mode)
    return action
