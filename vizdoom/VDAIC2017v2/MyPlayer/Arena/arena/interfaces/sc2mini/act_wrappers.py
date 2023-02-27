""" Gym env wrappers """
from gym import spaces
from arena.utils.unit_util import collect_units_by_alliance
from arena.utils.unit_util import find_units_by_tag
from arena.utils.constant import AllianceType
from arena.utils.dist_util import find_nearest
from arena.utils.unit_util import find_weakest
from arena.utils.action_util import move_pos
from arena.utils.action_util import hold_pos
from arena.utils.action_util import attack_unit, run_away_from_closest_enemy
from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace

import numpy as np


# 4 Move and 2 Attack Discrete actions wrapper
class Discre4M2AFunc(object):
    def __init__(self, units_orders):
        self.action_space = spaces.Tuple([spaces.Discrete(7)] * len(units_orders[0]))
        self.units_order = units_orders

    @staticmethod
    def action_map(unit, units, act):
        if act == 0:
            return hold_pos(unit)
        elif act <= 2:
            enemy = collect_units_by_alliance(units, AllianceType.ENEMY.value)
            target = find_nearest(enemy, unit) if act == 1 else find_weakest(enemy)
            if target:
                return attack_unit(unit, target)
            else:
                return None
        elif act <= 6:
            pos_x, pos_y = unit.pos.x, unit.pos.y
            dx, dy = [0, 0, -1, 1], [1, -1, 0, 0]
            pos = [pos_x + dx[act - 3], pos_y + dy[act - 3]]
            return move_pos(unit, pos)

    def action_transform(self, action, timestep):
        """action = [unit_action, ... , unit_action]"""
        raw_action = []
        units = timestep.observation.raw_data.units
        for unit_action, tag in zip(action, self.units_order[0]):
            unit = find_units_by_tag(units, tag)
            if unit:
                raw_unit_action = self.action_map(unit[0], units, unit_action)
                if raw_unit_action:
                    raw_action.append(raw_unit_action)
        return raw_action


class Discre4M2AInt(Interface):
    def __init__(self, interface):
        super(Discre4M2AInt, self).__init__(interface)
        self.wrapper = None

    def reset(self, obs, **kwargs):
        super(Discre4M2AInt, self).reset(obs, **kwargs)
        self.wrapper = Discre4M2AFunc(self.unwrapped().units_order)

    @property
    def action_space(self):
        if self.wrapper:
            return self.wrapper.action_space
        else:
            return NoneSpace()

    def act_trans(self, action):
        act = self.wrapper.action_transform(action, self.unwrapped()._obs)
        if self.inter:
            act = self.inter.act_trans(act)
        return act


class CombineActFunc(object):
    """ Combine multiple discrete action space as a whole """
    def __init__(self, inter):
        n_dim = 1
        ll = []
        for act_spa in inter.action_space.spaces:
            n_dim *= act_spa.n
        act_map = []
        for i in range(0,n_dim):
            dst = []
            j = i
            for act_spa in inter.action_space.spaces:
                dst.append(j % act_spa.n)
                j //= act_spa.n
            act_map.append(tuple(dst))
        self.act_map = act_map
        self.action_space = spaces.Discrete(n_dim)

    def action_transform(self, action, timestep):
        return self.act_map[action]


class CombineActInt(Interface):
    def __init__(self, interface):
        super(CombineActInt, self).__init__(interface)
        self.wrapper = None

    def reset(self, obs, **kwargs):
        super(CombineActInt, self).reset(obs, **kwargs)
        assert self.inter
        self.wrapper = CombineActFunc(self.inter)

    @property
    def action_space(self):
        if self.wrapper:
            return self.wrapper.action_space
        else:
            return NoneSpace()

    def act_trans(self, action):
        act = self.wrapper.action_transform(action, self.unwrapped()._obs)
        if self.inter:
            act = self.inter.act_trans(act)
        return act

# 8 Move and n Attack Discrete actions wrapper
class Discre8MnAFunc(object):
    def __init__(self, units_orders):
        self.action_space = spaces.Tuple([spaces.Discrete(9+len(units_orders[1]))] * len(units_orders[0]))
        self.units_order = units_orders
        self.n_enemy = len(self.units_order[1])

    def action_map(self, unit, units, act):
        if act == 0:
            return hold_pos(unit)
        elif act <= self.n_enemy:
            target = find_units_by_tag(units, self.units_order[1][act-1])
            if target:
                return attack_unit(unit, target[0])
            else:
                return None
        elif act <= 8 + self.n_enemy:
            pos_x, pos_y = unit.pos.x, unit.pos.y
            dx, dy = [0, 0, -1, 1, -1, 1, 1, -1], [1, -1, 0, 0, 1, 1, -1, -1]
            pos = [pos_x + dx[act - self.n_enemy - 1], pos_y + dy[act - self.n_enemy - 1]]
            return move_pos(unit, pos)

    def action_transform(self, action, timestep):
        """action = [unit_action, ... , unit_action]"""
        raw_action = []
        units = timestep.observation.raw_data.units
        for unit_action, tag in zip(action, self.units_order[0]):
            unit = find_units_by_tag(units, tag)
            if unit:
                raw_unit_action = self.action_map(unit[0], units, unit_action)
                if raw_unit_action:
                    raw_action.append(raw_unit_action)
        return raw_action


class Discre8MnAInt(Interface):
    def __init__(self, interface):
        super(Discre8MnAInt, self).__init__(interface)
        self.wrapper = None

    def reset(self, obs, **kwargs):
        super(Discre8MnAInt, self).reset(obs, **kwargs)
        self.wrapper = Discre8MnAFunc(self.unwrapped().units_order)

    @property
    def action_space(self):
        if self.wrapper:
            return self.wrapper.action_space
        else:
            return NoneSpace()

    def act_trans(self, action):
        act = self.wrapper.action_transform(action, self.unwrapped()._obs)
        if self.inter:
            act = self.inter.act_trans(act)
        return act


class MultiActInt(Interface):
    @property
    def action_space(self):
        #if self.inter.action_space is None: # TODO
        #    return None
        assert isinstance(self.inter.action_space, spaces.Tuple)
        nvec = []
        for space in self.inter.action_space.spaces:
            assert isinstance(space, spaces.Discrete)
            nvec.append(space.n)
        action_space = spaces.MultiDiscrete(nvec)
        return action_space
