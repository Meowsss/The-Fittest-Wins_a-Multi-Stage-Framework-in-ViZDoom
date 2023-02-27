"""Strategy Manager"""
import random

from pysc2.lib.typeenums import UNIT_TYPEID

from ..tstarbot_rules.combat_strategy.squad import Squad
from ..tstarbot_rules.combat_strategy.squad import SquadStatus, MutaliskSquadStatus
from ..tstarbot_rules.combat_strategy.army import Army
from ..tstarbot_rules.data.queue.combat_command_queue import CombatCmdType
from ..tstarbot_rules.data.queue.combat_command_queue import CombatCommand
from ..tstarbot_rules.data.pool.macro_def import COMBAT_UNITS, COMBAT_UNITS_FOOD_DICT
from ..tstarbot_rules.data.pool.macro_def import BUILDING_UNITS
from ..tstarbot_rules.data.pool.map_tool import get_slopes
from arena.interfaces.sc2full.rules.utils.com_str_utils import *


UNIT_BLACKLIST = {
    UNIT_TYPEID.ZERG_QUEEN.value,
    # UNIT_TYPEID.ZERG_MUTALISK.value
    # UNIT_TYPEID.ZERG_ZERGLING.value
}


class ZergSimpleComMgr(object):
    def __init__(self, dc):
        self._dc = dc
        self._army = Army()
        self._cmds = []
        self._global_step = 0

    def reset(self):
        self._army = Army()
        self._global_step = 0

    def update(self, dc, exe_cmd):
        self._army.update(dc.dd.combat_pool)
        self._dc = dc
        self._command_army(dc.dd.combat_command_queue, exe_cmd)

    def _command_army(self, cmd_queue, action):
        self._cmds = list()
        self._create_army_by_size()

        if action == 0:
            self._command_army_attack(cmd_queue)
        if action == 1:
            self._command_army_rally_base(cmd_queue)

    def _command_army_rally_base(self, cmd_queue):
        bases = self._dc.dd.base_pool.bases
        if len(bases) == 0:
            return False
        enemy_pool = self._dc.dd.enemy_pool
        enemy_combat_units = find_enemy_combat_units(enemy_pool.units)
        if len(enemy_combat_units) == 0 or enemy_pool.closest_cluster is None:
            enemy_pos = {'x': self._dc.dd.base_pool.enemy_home_pos[0],
                         'y': self._dc.dd.base_pool.enemy_home_pos[1]}
        else:
            closest_enemy = None
            d_min = 100000
            for tag in bases:
                b = bases[tag].unit
                for e in enemy_combat_units:
                    d_eb = cal_square_dist(e, b)
                    if d_eb < d_min:
                        d_min = d_eb
                        closest_enemy = e
            enemy_pos = {'x': closest_enemy.float_attr.pos_x,
                         'y': closest_enemy.float_attr.pos_y}
        self._danger_base_pos = find_base_pos_in_danger(self._dc, enemy_pos)
        self._rally_pos = {'x': self._danger_base_pos['x'] * 0.9 + enemy_pos['x'] * 0.1,
                           'y': self._danger_base_pos['y'] * 0.9 + enemy_pos['y'] * 0.1}
        if self._rally_pos is None:
            return None
        for squad in self._army.squads:
            if squad.uniform or squad.status == SquadStatus.SCOUT:
                continue
            squad.status = SquadStatus.MOVE
            cmd = CombatCommand(
                type=CombatCmdType.MOVE,
                squad=squad,
                position=self._rally_pos)
            cmd_queue.push(cmd)
            self._cmds.append(cmd)

        rallied_squads = [squad for squad in self._army.squads
                          if distance(squad.centroid, self._rally_pos) < 10]
        for squad in rallied_squads:
            if squad.uniform or squad.status == SquadStatus.SCOUT:
                continue
            squad.status = SquadStatus.IDLE

    def _command_army_attack(self, cmd_queue):
        # attack
        target_pos = {'x': self._dc.dd.base_pool.enemy_home_pos[0],
                      'y': self._dc.dd.base_pool.enemy_home_pos[1]}
        for squad in self._army.squads:
            squad.status = SquadStatus.ATTACK
            cmd = CombatCommand(
                type=CombatCmdType.ATTACK,
                squad=squad,
                position=target_pos)
            cmd_queue.push(cmd)
            self._cmds.append(cmd)

    def _create_army_by_size(self, size=1):
        self._create_fixed_size_exclude_type_squads(size, UNIT_BLACKLIST)

    def _create_fixed_size_exclude_type_squads(self, squad_size, unit_type_blacklist=set()):
        allowed_unsquaded_units = [u for u in self._army.unsquaded_units if
                                   u.unit.unit_type not in unit_type_blacklist]
        while len(allowed_unsquaded_units) >= squad_size:
            self._army.create_squad(
                random.sample(allowed_unsquaded_units, squad_size))
            allowed_unsquaded_units = [
                u for u in self._army.unsquaded_units
                if u.unit.unit_type not in unit_type_blacklist
            ]
