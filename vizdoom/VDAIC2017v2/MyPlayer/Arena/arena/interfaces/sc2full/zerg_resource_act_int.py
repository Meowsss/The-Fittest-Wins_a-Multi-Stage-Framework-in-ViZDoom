""" Interface wrappers """
from gym import spaces
from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace
from .tstarbot_rules.resource.resource_mgr import ZergResourceMgr
from .tstarbot_rules.act.act_mgr import ActMgr
from .tstarbot_rules.production_strategy.build_cmd import *


class ZergResourceActInt(Interface):
    def __init__(self, inter, append=True, auto_resource=True, step_mul=1):
        super(ZergResourceActInt, self).__init__(inter)
        self.append = append
        self.auto_resource = auto_resource
        self.resource_mgr = ZergResourceMgr(self.unwrapped().dc)
        self.am_mgr = ActMgr()
        self.step_mul = step_mul
        self._actions = [
            'none',
            'gas_first',
            'mineral_first',
        ]

    def reset(self, obs, **kwargs):
        super(ZergResourceActInt, self).reset(obs, **kwargs)
        self.resource_mgr.reset()
        self.unwrapped().mask_size += len(self._actions)

    @property
    def action_space(self):
        if self.append:
            if isinstance(self.inter.action_space, NoneSpace):
                return NoneSpace()
            assert isinstance(self.inter.action_space, spaces.MultiDiscrete)
            return spaces.MultiDiscrete(list(self.inter.action_space.nvec) +
                                        [len(self._actions)] * (1 - self.auto_resource))
        else:
            return spaces.MultiDiscrete([len(self._actions)] * (1 - self.auto_resource))

    @property
    def inter_action_len(self):
        return len(self.inter.action_space.nvec)

    def act_trans(self, action):
        if self.append:
            act_old = super(ZergResourceActInt, self).act_trans(action[0:self.inter_action_len])
            act = action[self.inter_action_len:]
        else:
            super(ZergResourceActInt, self).act_trans([])
            act_old = []
            act = action
        if (self.unwrapped().steps - 1) % self.step_mul != 0:
            act = [0]
        if self.auto_resource:
            if self.should_gas_first(self.unwrapped().dc.sd.obs):
                self.unwrapped().dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=True))
            if self.should_mineral_first(self.unwrapped().dc.sd.obs):
                self.unwrapped().dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=False))
        else:
            if self._actions[act[0]] == 'gas_first':
                self.unwrapped().dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=True))
            elif self._actions[act[0]] == 'mineral_first':
                self.unwrapped().dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=False))
        self.resource_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        raw_actions = self.am_mgr.pop_actions()
        return act_old + raw_actions

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if not self.auto_resource:
            if self.unwrapped().steps % self.step_mul == 0:
                action_avail = [1] * len(self._actions)
            else:
                action_avail = [1] + [0] * (len(self._actions) - 1)
            self.unwrapped().mask.extend(action_avail)
        return obs

    @staticmethod
    def should_gas_first(obs):
        play_info = obs["player"]
        minerals, vespene = play_info[1:3]
        if minerals > 400 and vespene < minerals / 3:
            return True
        return False

    @staticmethod
    def should_mineral_first(obs):
        play_info = obs["player"]
        minerals, vespene = play_info[1:3]
        if vespene > 300 and minerals < 2 * vespene:
            return True
        return False