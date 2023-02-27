""" Gym env wrappers """
from gym import spaces
import numpy as np
from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace
from arena.interfaces.sc2full.rules.zerg_com_str_rules import ZergStrategyMgr
from arena.interfaces.sc2full.rules.utils.com_str_utils import COM_CMD_TYPE
from .tstarbot_rules.combat.combat_mgr import ZergCombatMgr
from .tstarbot_rules.act.act_mgr import ActMgr


class ZergCombatActInt(Interface):
    def __init__(self, inter, append=True, sub_actions=None, step_mul=1):
        super(ZergCombatActInt, self).__init__(inter)
        self.append = append
        self.str_mgr = ZergStrategyMgr(self.unwrapped().dc)
        self.micro_mgr = ZergCombatMgr(self.unwrapped().dc)
        self.am_mgr = ActMgr()
        self.areas = None
        self.area_num = 0
        self.last_act_id = None
        self.sub_actions = sub_actions
        self.step_mul = step_mul

    def reset(self, obs, **kwargs):
        super(ZergCombatActInt, self).reset(obs, **kwargs)
        self.str_mgr.reset()
        self.micro_mgr.reset()
        dc = self.unwrapped().dc
        areas = dc.dd.base_pool.resource_cluster
        # sort areas here
        home_dist_dict = dc.dd.base_pool.enemy_home_dist
        dists = [home_dist_dict[areas[i]] for i in range(len(areas))]
        area_sort_id = np.argsort(dists)
        self.areas = [areas[i] for i in area_sort_id]
        self.area_num = len(self.areas)
        self.last_act_id = None
        if self.sub_actions is not None:
            assert all([0 <= a < self.area_num + 4 for a in self.sub_actions])
        else:
            self.sub_actions = range(0, self.area_num + 4)
        self.unwrapped().mask_size += len(self.sub_actions) + 2

    @property
    def action_space(self):
        if self.areas is None:
            return NoneSpace()
        com_action_space = spaces.MultiDiscrete([len(self.sub_actions), 2])
        if self.append:
            assert isinstance(self.inter.action_space, spaces.MultiDiscrete)
            return spaces.MultiDiscrete(list(self.inter.action_space.nvec) +
                                        list(com_action_space.nvec))
        else:
            return com_action_space

    @property
    def inter_action_len(self):
        return len(self.inter.action_space.nvec)

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if self.unwrapped().steps % self.step_mul == 0:
            self.unwrapped().mask.extend([1] * (len(self.sub_actions) + 2))
        else:
            self.unwrapped().mask.extend([1] + [0] * (len(self.sub_actions) - 1) + [0, 1])
        return obs

    def act_trans(self, action):
        if self.append:
            act_pre_int = super(ZergCombatActInt, self).act_trans(action[0:self.inter_action_len])
            com_act_id = action[self.inter_action_len:]
        else:
            super(ZergCombatActInt, self).act_trans([])
            act_pre_int = []
            com_act_id = action[0:]
        if (self.unwrapped().steps - 1) % self.step_mul != 0:
            com_act_id = [0, 1]
        if com_act_id[1] == 0:
            cmd = self._convert_to_cmd(com_act_id[0])
            self.last_act_id = com_act_id[0]
        else:  # continue last cmd
            if self.last_act_id is None:
                cmd = (None, None)
            else:
                cmd = self._convert_to_cmd(self.last_act_id)
        self.str_mgr.update(dc=self.unwrapped().dc, exe_cmd=cmd)
        self.micro_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        raw_actions = self.am_mgr.pop_actions()
        # print('RawAction num: {}'.format([len(act_pre_int), len(raw_actions)]))
        return act_pre_int + raw_actions

    def _convert_to_cmd(self, com_act_id):
        com_act_id = self.sub_actions[com_act_id]
        if com_act_id in range(0, self.area_num):
            cmd_type = COM_CMD_TYPE.ATK
            target_area = self.areas[com_act_id]
            target_area = target_area.ideal_base_pos
            target_area = {'x': target_area[0], 'y': target_area[1]}
        elif com_act_id == self.area_num:
            cmd_type = COM_CMD_TYPE.ATK
            target_area = None
        elif com_act_id == self.area_num + 1:
            cmd_type = COM_CMD_TYPE.DEF
            target_area = None
        elif com_act_id == self.area_num + 2:
            cmd_type = COM_CMD_TYPE.RAL_BASE
            target_area = None
        elif com_act_id == self.area_num + 3:
            cmd_type = COM_CMD_TYPE.RAL_BEFORE_ATK
            target_area = None
        # else:
        #     cmd_type = None
        #     target_area = None
        # elif com_act_id == self.area_num + 1:
        #     # print('cmd defend')
        #     cmd_type = COM_CMD_TYPE.DEF
        #     target_area = None
        # elif com_act_id == self.area_num + 2:
        #     # print('cmd harass')
        #     cmd_type = COM_CMD_TYPE.HAR
        #     target_area = None
        # else:
        #     # print('cmd rock')
        #     cmd_type = COM_CMD_TYPE.ROC
        #     target_area = None
        cmd = (cmd_type, target_area)
        return cmd
