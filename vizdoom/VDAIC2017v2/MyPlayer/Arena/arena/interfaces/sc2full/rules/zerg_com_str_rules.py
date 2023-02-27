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


class ZergStrategyMgr(object):
    def __init__(self, dc):
        self._dc = dc
        self._enable_render = False
        self._army = Army()
        self._cmds = []
        self._ready_to_go = False
        self._ready_to_attack = False
        self._mutalisk_ready_to_go = False
        self._mutalisk_ready_to_harass = False
        self._food_trigger = 10

        self._rally_pos = None
        self._rally_pos_for_attack = None
        self._verbose = 0

        self._global_step = 0
        self._slopes = None

    def reset(self):
        self._army = Army()
        self._ready_to_go = False
        self._ready_to_attack = False
        self._mutalisk_ready_to_go = False
        self._mutalisk_ready_to_harass = False
        self._food_trigger = 100

        self._rally_pos = None
        self._rally_pos_for_attack = None

        self._global_step = 0
        self._slopes = None

    def update(self, dc, exe_cmd):
        self._army.update(dc.dd.combat_pool)
        self._dc = dc
        if self._global_step == 1:
            self._slopes = get_slopes(dc.sd.timestep, dc.game_version)
        self._command_army(dc.dd.combat_command_queue, exe_cmd)
        self._global_step += 1

    def _command_army(self, cmd_queue, exe_cmd):
        self._cmds = list()
        self._create_army_by_size()

        if exe_cmd[0] == COM_CMD_TYPE.ATK:
            self._command_army_attack(cmd_queue, exe_cmd[1])
        if exe_cmd[0] == COM_CMD_TYPE.RAL_BASE:
            self._command_army_rally_base(cmd_queue)
        if exe_cmd[0] == COM_CMD_TYPE.RAL_BEFORE_ATK:
            self._command_army_rally_before_atk(cmd_queue)
        if exe_cmd[0] == COM_CMD_TYPE.DEF:
            self._command_army_defend(cmd_queue)
        if exe_cmd[0] == COM_CMD_TYPE.HAR:
            self._command_army_harass(cmd_queue)
        if exe_cmd[0] == COM_CMD_TYPE.ROC:
            self._command_army_atk_stone(cmd_queue)
        if exe_cmd[0] is None:
            pass

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


    def _command_army_rally_before_atk(self, cmd_queue):
        bases = self._dc.dd.base_pool.bases
        if len(bases) == 0:
            return False
        enemy_pool = self._dc.dd.enemy_pool
        enemy_combat_units = find_enemy_combat_units(enemy_pool.units)
        if len(enemy_combat_units) == 0 or enemy_pool.closest_cluster is None:
            enemy_pos = {'x': self._dc.dd.base_pool.enemy_home_pos[0],
                         'y': self._dc.dd.base_pool.enemy_home_pos[1]}
        elif enemy_pool.closest_cluster:
            enemy_pos = enemy_pool.closest_cluster.centroid
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
        self._rally_pos = {'x': self._danger_base_pos['x'] * 0.3 + enemy_pos['x'] * 0.7,
                           'y': self._danger_base_pos['y'] * 0.3 + enemy_pos['y'] * 0.7}
        if self._rally_pos is None:
            return None
        for squad in self._army.squads:
            if squad.uniform or squad.status == SquadStatus.SCOUT:
                continue
            squad.status = SquadStatus.ATTACK
            cmd = CombatCommand(
                type=CombatCmdType.ATTACK,
                squad=squad,
                position=self._rally_pos)
            cmd_queue.push(cmd)
            self._cmds.append(cmd)


    def _command_army_attack(self, cmd_queue, target_pos):
        # attack
        if self._verbose > 0:
            if target_pos is None:
                print('All attack nearest enemy')
            else:
                print('Attack {}'.format(target_pos))
        for squad in self._army.squads:
            if squad.uniform or squad.status == SquadStatus.SCOUT:
                continue
            squad.status = SquadStatus.ATTACK
            cmd = CombatCommand(
                type=CombatCmdType.ATTACK,
                squad=squad,
                position=target_pos)
            cmd_queue.push(cmd)
            self._cmds.append(cmd)

    def _command_army_harass(self, cmd_queue):
        poses = get_mutalisk_safe_pos(self._dc)
        if poses:
            self._mutalisk_harass(cmd_queue, 'main_base', poses[0], poses[1])
            self._mutalisk_harass(cmd_queue, 'sub_base1', poses[4], poses[5])

    def _command_army_defend(self, cmd_queue):
        enemy_pool = self._dc.dd.enemy_pool
        enemy_combat_units = find_enemy_combat_units(enemy_pool.units)
        if len(enemy_combat_units) == 0 or enemy_pool.closest_cluster is None:
            return False
        bases = self._dc.dd.base_pool.bases

        if 2 < len(bases) <= 16:
            enemy_attacking_me = False
            closest_enemy = None
            for tag in bases:
                b = bases[tag].unit
                for e in enemy_combat_units:
                    if cal_square_dist(e, b) < 30:
                        enemy_attacking_me = True
                        closest_enemy = e
                        break
                if enemy_attacking_me:
                    break

            if enemy_attacking_me:
                if self._verbose > 0:
                    print('Defend.')
                for squad in self._army.squads + [self._create_queen_squads()]:
                    if squad.uniform is not None and \
                            squad.combat_status not in [MutaliskSquadStatus.IDLE,
                                                        MutaliskSquadStatus.PHASE1]:
                        continue
                    if squad.status == SquadStatus.SCOUT:
                        continue
                    squad.status = SquadStatus.MOVE
                    cmd = CombatCommand(
                        type=CombatCmdType.ATTACK,
                        squad=squad,
                        position={
                            'x': closest_enemy.float_attr.pos_x,
                            'y': closest_enemy.float_attr.pos_y
                        })
                    cmd_queue.push(cmd)
                    self._cmds.append(cmd)
                return True
        elif len(bases) <= 2:
            enemy_attacking_me = False
            for tag in bases:
                b = bases[tag].unit
                for e in enemy_combat_units:
                    if cal_square_dist(e, b) < 60:
                        enemy_attacking_me = True
                        # closest_enemy = e
                        break
                if enemy_attacking_me:
                    break

            if enemy_attacking_me:
                if self._verbose > 0:
                    print('Defend rush.')
                for squad in self._army.squads + [self._create_queen_squads()]:
                    if squad.uniform is not None and \
                            squad.combat_status not in [MutaliskSquadStatus.IDLE,
                                                        MutaliskSquadStatus.PHASE1]:
                        continue
                    if squad.status == SquadStatus.SCOUT:
                        continue
                    squad.status = SquadStatus.MOVE

                    second_base_pos = get_second_base_pos(self._dc)
                    defend_base_pos = get_main_base_pos(self._dc) \
                        if second_base_pos is None else second_base_pos

                    if squad.uniform == 'queen':
                        cmd = CombatCommand(
                            type=CombatCmdType.ATTACK,
                            squad=squad,
                            position=get_slope_up_pos(self._slopes, defend_base_pos, 0.5)
                        )
                    else:
                        cmd = CombatCommand(
                            type=CombatCmdType.ATTACK,
                            squad=squad,
                            position=get_slope_up_pos(self._slopes, defend_base_pos, 0.3)
                        )

                    cmd_queue.push(cmd)
                    self._cmds.append(cmd)
                return True

        return False

    def _command_army_atk_stone(self, cmd_queue):
        rocks = [u for u in self._dc.sd.obs['units']
                 if u.int_attr.unit_type == UNIT_TYPEID.NEUTRAL_DESTRUCTIBLEROCKEX1DIAGONALHUGEBLUR.value]
        home_pos = {'x': self._dc.dd.base_pool.home_pos[0],
                    'y': self._dc.dd.base_pool.home_pos[1]}

        if len(rocks) > 1:
            r0_pos = {'x': rocks[0].float_attr.pos_x,
                      'y': rocks[0].float_attr.pos_y}
            r1_pos = {'x': rocks[1].float_attr.pos_x,
                      'y': rocks[1].float_attr.pos_y}
            if distance(r0_pos, home_pos) > distance(r1_pos, home_pos):
                enemy_rock = rocks[0]
            else:
                enemy_rock = rocks[1]
        else:
            return False

        if self._verbose > 0:
            print('Attack rock.')
        for squad in self._army.squads:
            if squad.uniform or squad.status == SquadStatus.SCOUT:
                continue
            squad.status = SquadStatus.ROCK
            cmd = CombatCommand(
                type=CombatCmdType.ROCK,
                squad=squad,
                position={'x': enemy_rock.float_attr.pos_x,
                          'y': enemy_rock.float_attr.pos_y}
            )
            cmd_queue.push(cmd)
            self._cmds.append(cmd)
        return True

    def _create_army_by_size(self, size=1):
        self._create_fixed_size_exclude_type_squads(size, UNIT_BLACKLIST)
        self._create_fixed_size_mutalisk_squads(squad_size=3, mutalisk_uniform='main_base', unique=True)
        self._create_fixed_size_mutalisk_squads(squad_size=3, mutalisk_uniform='sub_base1', unique=True)
        self._create_fixed_size_mutalisk_squads(squad_size=3, mutalisk_uniform='sub_base2', unique=True)

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

    def _create_fixed_size_mutalisk_squads(self, squad_size, mutalisk_uniform, unique=False):
        if unique:
            squads = [s for s in self._army.squads]
            if mutalisk_uniform in [s.uniform for s in squads]:
                return None
        unsquaded_mutalisks = [u for u in self._army.unsquaded_units if
                               u.unit.unit_type == UNIT_TYPEID.ZERG_MUTALISK.value]
        while len(unsquaded_mutalisks) >= squad_size:
            if self._verbose > 0:
                print('Create mutalisk squad: ' + mutalisk_uniform)
            self._army.create_squad(
                random.sample(unsquaded_mutalisks, squad_size), mutalisk_uniform)
            unsquaded_mutalisks = [
                u for u in self._army.unsquaded_units
                if u.unit.unit_type == UNIT_TYPEID.ZERG_MUTALISK.value]

    def _create_queen_squads(self):
        queens = [u for u in self._army.unsquaded_units
                  if u.unit.unit_type == UNIT_TYPEID.ZERG_QUEEN.value]
        queen_squad = Squad(queens, 'queen')
        return queen_squad

    def _estimate_enemy_army_power(self):
        enemy_combat_units = [u for u in self._dc.dd.enemy_pool.units if
                              u.int_attr.unit_type in COMBAT_UNITS]
        enemy_army_food = 0
        for e in enemy_combat_units:
            if e.int_attr.unit_type not in COMBAT_UNITS_FOOD_DICT.keys():
                continue
            enemy_army_food += COMBAT_UNITS_FOOD_DICT[e.int_attr.unit_type]
        return enemy_army_food

    def _estimate_self_army_power_in_battle(self):
        enemy_combat_units = [u for u in self._dc.dd.enemy_pool.units if
                              u.int_attr.unit_type in COMBAT_UNITS]
        if len(enemy_combat_units) == 0:
            return -1
        # do not include mutalisk
        self_combat_units = [u.unit for u in self._dc.dd.combat_pool.units
                             if u.unit.int_attr.unit_type !=
                             UNIT_TYPEID.ZERG_MUTALISK.value]
        battle_pos = None
        for e in enemy_combat_units:
            for u in self_combat_units:
                if cal_square_dist(u, e) < 30:
                    battle_pos = {'x': u.float_attr.pos_x,
                                  'y': u.float_attr.pos_y}

        if battle_pos is None:
            return -1

        self_combat_units_in_battle = [u for u in self_combat_units
                                       if distance({'x': u.float_attr.pos_x,
                                                    'y': u.float_attr.pos_y}, battle_pos) < 10]
        self_army_food_in_battle = 0
        for u in self_combat_units_in_battle:
            if u.int_attr.unit_type not in COMBAT_UNITS_FOOD_DICT.keys():
                continue
            self_army_food_in_battle += COMBAT_UNITS_FOOD_DICT[u.int_attr.unit_type]
        return self_army_food_in_battle

    def _mutalisk_harass(self, cmd_queue, mutalisk_uniform, harass_station_pos1, harass_station_pos2):
        enemy_units = self._dc.dd.enemy_pool.units
        enemy_buildings_and_drones_overlords = [e for e in enemy_units
                                                if e.int_attr.unit_type in
                                                BUILDING_UNITS or
                                                e.int_attr.unit_type in [
                                                    UNIT_TYPEID.ZERG_OVERLORD.value,
                                                    UNIT_TYPEID.ZERG_DRONE.value,
                                                    UNIT_TYPEID.ZERG_OVERSEER.value]
                                                ]

        for squad in self._army.squads:
            if squad.uniform == mutalisk_uniform and \
                    squad.combat_status == MutaliskSquadStatus.IDLE:
                # rally at a safe corner
                cmd = CombatCommand(
                    type=CombatCmdType.ATTACK,  # for defend
                    squad=squad,
                    position=harass_station_pos1)
                cmd_queue.push(cmd)
                self._cmds.append(cmd)

        rallied_mutalisk_squads = [squad for squad in self._army.squads
                                   if squad.uniform == mutalisk_uniform and
                                   distance(squad.centroid, harass_station_pos1) < 5]
        if len(rallied_mutalisk_squads) > 0:
            rallied_mutalisk_squads[0].combat_status = MutaliskSquadStatus.PHASE1

        for squad in self._army.squads:
            if squad.uniform == mutalisk_uniform and \
                    squad.combat_status == MutaliskSquadStatus.PHASE1:
                cmd = CombatCommand(
                    type=CombatCmdType.MOVE,
                    squad=squad,
                    position=harass_station_pos2)
                cmd_queue.push(cmd)
                self._cmds.append(cmd)

        rallied_mutalisk_squads = [squad for squad in self._army.squads
                                   if squad.uniform == mutalisk_uniform and
                                   distance(squad.centroid, harass_station_pos2) < 5]
        if len(rallied_mutalisk_squads) > 0:
            rallied_mutalisk_squads[0].combat_status = MutaliskSquadStatus.PHASE2

        if len(enemy_buildings_and_drones_overlords) == 0:
            return None
        for squad in self._army.squads:
            if squad.uniform == mutalisk_uniform and \
                    squad.combat_status == MutaliskSquadStatus.PHASE2:
                closest_enemy_base = find_closest_enemy_to_pos(squad.centroid,
                                                               enemy_buildings_and_drones_overlords)
                cmd = CombatCommand(
                    type=CombatCmdType.ATTACK,
                    squad=squad,
                    position={'x': closest_enemy_base.float_attr.pos_x,
                              'y': closest_enemy_base.float_attr.pos_y})
                cmd_queue.push(cmd)
                self._cmds.append(cmd)
