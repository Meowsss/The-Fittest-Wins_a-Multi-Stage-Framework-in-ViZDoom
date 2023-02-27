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

import numpy as np

class ImgActInt(Interface):
    def __init__(self, inter):
        super(self.__class__, self).__init__(inter)

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def action_space(self):
        return spaces.Box(low=0, high=1, shape=(32, 32, 2), dtype=int)

    def act_trans(self, action):
        act_2d = action.argmax(axis=-1)
        tag_2d = self.inter.self_tag_2d

        raw_actions = []
        nnd_idx = np.nonzero(tag_2d)
        for x, y in zip(nnd_idx[0], nnd_idx[1]):
            tag = tag_2d[x, y]
            act = act_2d[x, y]
            raw_actions.append(self._discrete_action(tag, act))
        if self.inter:
            raw_actions = self.inter.act_trans(raw_actions)
        return raw_actions

    def _discrete_action(self, unit_tag, act_id):
        timestep = self.unwrapped()._obs
        units = timestep.observation.raw_data.units
        enemies = collect_units_by_alliance(units, AllianceType.ENEMY.value)
        u = find_units_by_tag(units, unit_tag)
        if not u:
            return []
        u = u[0]

        if len(enemies) == 0:
            # print('warning: no enemy found!')
            return hold_pos(u)

        target = find_nearest(enemies, u)

        if act_id == 0:
            action = attack_unit(u, target)
        else:
            action = run_away_from_closest_enemy(u, target)
        return action

