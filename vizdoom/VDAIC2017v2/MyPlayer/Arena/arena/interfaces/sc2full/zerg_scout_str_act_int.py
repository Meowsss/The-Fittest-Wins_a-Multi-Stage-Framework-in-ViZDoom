""" Gym env wrappers """
from gym import spaces
from arena.interfaces.interface import Interface
from arena.interfaces.sc2full.rules.zerg_scout_str_rules import ZergScoutMicroMgr
from .tstarbot_rules.act.act_mgr import ActMgr

class ZergScoutActInt(Interface):
    def __init__(self, inter, append=True):
        super(ZergScoutActInt, self).__init__(inter)
        self.append = append
        self.micro_mgr = ZergScoutMicroMgr()
        self.am_mgr = ActMgr()
        self.area_num = 0

    def reset(self, obs, **kwargs):
        super(ZergScoutActInt, self).reset(obs, **kwargs)
        dc = self.unwrapped().dc
        self.area_num = dc.dd.scout_pool.scout_base_target_num()

    @property
    def action_space(self):
        if 0 == self.area_num:
            return self.inter.action_space
        scout_action_space = spaces.MultiDiscrete([self.area_num])
        if self.append:
            assert isinstance(self.inter.action_space, spaces.MultiDiscrete)
            return spaces.MultiDiscrete(list(self.inter.action_space.nvec) +
                                        list(scout_action_space.nvec))
        else:
            return scout_action_space

    @property
    def inter_action_len(self):
        return len(self.inter.action_space.nvec)

    def act_trans(self, action):
        if self.append:
            act_pre_int = super(ZergScoutActInt, self).act_trans(action[0:self.inter_action_len])
            com_act_id = action[self.inter_action_len:]
        else:
            act_pre_int = []
            com_act_id = action[0:]

        self.micro_mgr.update(self.unwrapped().dc, 
                              self.am_mgr, 
                              self._convert_to_cmd(com_act_id))
        raw_actions = self.am_mgr.pop_actions()
        return act_pre_int + raw_actions

    def _convert_to_cmd(self, com_act_id):
        dc = self.unwrapped().dc
        target = dc.dd.scout_pool.get_scout_target_by_num(com_act_id[0])
        return target


