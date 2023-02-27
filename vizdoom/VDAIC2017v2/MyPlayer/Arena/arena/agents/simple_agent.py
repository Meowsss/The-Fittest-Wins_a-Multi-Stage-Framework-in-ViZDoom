"""
Test agent to control marines
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from arena.utils.unit_util import collect_units_by_alliance
from arena.utils.constant import AllianceType
from arena.utils.dist_util import find_nearest, distance
from arena.utils.unit_util import find_weakest, find_strongest
from arena.utils.action_util import attack_unit, run_away_from_closest_enemy
from arena.agents.base_agent import BaseAgent
import gym


class RandomAgent(BaseAgent):
    """Random action agent."""
    def __init__(self, action_space=None):
        super(RandomAgent, self).__init__()
        self.action_space = action_space

    def step(self, obs):
        super(RandomAgent, self).step(obs)
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        else:
            return None

    def reset(self, timestep=None):
        super(RandomAgent, self).reset(timestep)
        assert isinstance(self.action_space, gym.Space)


class AtkNearestAgent(BaseAgent):
    """An agent that attack nearest enemy."""

    def __init__(self):
        super(AtkNearestAgent, self).__init__()

    def step(self, timestep):
        super(AtkNearestAgent, self).step(timestep)
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
        enemy = collect_units_by_alliance(units, AllianceType.ENEMY.value)
        actions = []
        for my_unit in my_units:
            target = find_nearest(enemy, my_unit)
            if target is not None:
                actions.append(attack_unit(my_unit, target))
        return actions

    def reset(self, timestep=None):
        super(AtkNearestAgent, self).reset(timestep)


class AtkWeakestAgent(BaseAgent):
    """An agent that attack weakest enemy."""

    def __init__(self):
        super(AtkWeakestAgent, self).__init__()

    def step(self, timestep):
        super(AtkWeakestAgent, self).step(timestep)
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
        enemy = collect_units_by_alliance(units, AllianceType.ENEMY.value)
        actions = []
        for my_unit in my_units:
            target = find_weakest(enemy)
            if target is not None:
                actions.append(attack_unit(my_unit, target))
        return actions

    def reset(self, timestep=None):
        super(AtkWeakestAgent, self).reset(timestep)


class AtkRunAgent(BaseAgent):
    """An agent that attack nearest enemy."""

    def __init__(self):
        super(AtkRunAgent, self).__init__()
        self.Immortal_range = 6.0 + 0.75 * 2

    def step(self, timestep):
        super(AtkRunAgent, self).step(timestep)
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
        enemy = collect_units_by_alliance(units, AllianceType.ENEMY.value)
        actions = []
        my_strongest_unit = find_strongest(my_units)
        for my_unit in my_units:
            target = find_nearest(enemy, my_unit)
            if target is not None:
                if distance(target, my_unit) < self.Immortal_range and \
                    float(my_unit.health) / my_unit.health_max < 0.3 and \
                        my_strongest_unit.health > my_unit.health:
                    if my_unit.health > target.health:
                        actions.append(attack_unit(my_unit, target))
                    else:
                        actions.append(run_away_from_closest_enemy(my_unit, target, 0.2))
                else:
                    actions.append(attack_unit(my_unit, target))
        return actions

    def act(self, timestep):
        v = 0
        p = 0
        return self.step(timestep), v, p


class RunAgent(BaseAgent):
    """An agent that always run away from closest enemy."""

    def __init__(self):
        super(RunAgent, self).__init__()
        self.Immortal_range = 6.0 + 0.75 * 2

    def step(self, timestep):
        super(RunAgent, self).step(timestep)
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
        enemy = collect_units_by_alliance(units, AllianceType.ENEMY.value)
        actions = []
        my_strongest_unit = find_strongest(my_units)
        for my_unit in my_units:
            target = find_nearest(enemy, my_unit)
            if target is not None:
                actions.append(run_away_from_closest_enemy(my_unit, target, 0.2))
        return actions
