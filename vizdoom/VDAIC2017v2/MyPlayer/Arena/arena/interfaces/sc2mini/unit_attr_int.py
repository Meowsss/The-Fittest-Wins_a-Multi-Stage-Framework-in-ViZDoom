""" Gym env wrappers """
from gym import spaces
from arena.utils.unit_util import find_units_by_tag
import numpy as np
from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace


def inv_dist(u, v):
    if v:
        x = (u.pos.x - v.pos.x)
        y = (u.pos.y - v.pos.y)
        r = (x * x + y * y) / (16 * 16 + 16 * 16)
        return 1 / (1 + r)
    else:
        return 0


# Extract Units' Attributes Wrapper
class UnitAttrFunc(object):
    def __init__(self, units_order, override, space_old):
        num_of_units = sum([len(camp) for camp in units_order])
        self.dimension = 10 + num_of_units
        observation_space = [spaces.Box(low=-1, high=2, dtype=np.float32,
                                        shape=(sum([len(units) for units in units_order]), self.dimension))]
        self.override = override
        if self.override or isinstance(space_old, NoneSpace):
            self.observation_space = spaces.Tuple(observation_space)
        else:
            self.observation_space = \
                spaces.Tuple(space_old.spaces + observation_space)
        self.units_order = units_order

    def basic_attr(self, u, lu):
        if not u:
            return np.array([0] * self.dimension, dtype=np.float32)
        else:
            l = [inv_dist(u, v) for v in lu]
            ret = np.array([1,
                            u.health / 100.0,
                            u.shield / 100.0,
                            u.energy / 100.0,
                            u.health / u.health_max if u.health_max > 0 else 0,
                            u.shield / u.shield_max if u.shield_max > 0 else 0,
                            u.energy / u.energy_max if u.energy_max > 0 else 0,
                            u.weapon_cooldown / 100.0,
                            (u.pos.x - 16.0) / 8.0,
                            (u.pos.y - 16.0) / 8.0,
                            ] + l, dtype=np.float32)
            assert (len(ret) == self.dimension)
            return ret

    def observation_transform(self, obs, timestep):
        units_vec = []
        units = timestep.observation.raw_data.units
        list_unit = []
        for units_order in self.units_order:
            for tag in units_order:
                unit = find_units_by_tag(units, tag)
                if unit:
                    list_unit.append(unit[0])
                else:
                    list_unit.append(None)
        for units_order in self.units_order:
            for tag in units_order:
                unit = find_units_by_tag(units, tag)
                if unit:
                    unit_vec = self.basic_attr(unit[0], list_unit)
                else:
                    unit_vec = self.basic_attr(None, list_unit)
                units_vec.append(unit_vec)
        observation = [np.array(units_vec)]
        return observation if self.override else list(obs) + observation


class UnitAttrInt(Interface):
    def __init__(self, interface, override=False):
        super(UnitAttrInt, self).__init__(interface)
        self.wrapper = None
        self.override = override

    def reset(self, obs, **kwargs):
        super(UnitAttrInt, self).reset(obs, **kwargs)
        self.wrapper = UnitAttrFunc(self.unwrapped().units_order, override=self.override,
                                    space_old=self.inter.observation_space)

    @property
    def observation_space(self):
        if self.wrapper:
            return self.wrapper.observation_space
        else:
            return NoneSpace()

    def obs_trans(self, obs):
        if self.inter:
            obs = self.inter.obs_trans(obs)
        obs = self.wrapper.observation_transform(obs, self.unwrapped()._obs)
        return obs
