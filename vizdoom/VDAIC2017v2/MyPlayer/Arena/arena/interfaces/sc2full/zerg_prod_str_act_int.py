""" Interface wrappers """
from gym import spaces
from arena.utils.spaces import NoneSpace
from arena.interfaces.interface import Interface
from arena.interfaces.sc2full.rules.zerg_prod_str_rules import ZergProductionMgr, ZergProdMgr
from .tstarbot_rules.building.building_mgr import ZergBuildingMgr
from .tstarbot_rules.act.act_mgr import ActMgr


class ZergProdActInt(Interface):
    def __init__(self, inter, append=True, auto_larva=True,
                 auto_pre=True, auto_supply=True, keep_order=False,
                 action_mask=False, opening_rule=None, step_mul=1):
        super(ZergProdActInt, self).__init__(inter)
        self.append = append
        if opening_rule:
            self.prodstr_mgr = ZergProdMgr(self.unwrapped().dc, opening=opening_rule,
                                           auto_larva=auto_larva, auto_supply=auto_supply)
        else:
            self.prodstr_mgr = ZergProductionMgr(self.unwrapped().dc,
                                                 auto_larva=auto_larva, auto_pre=auto_pre,
                                                 auto_supply=auto_supply, keep_order=keep_order)
        self.build_mgr = ZergBuildingMgr(self.unwrapped().dc)
        self.am_mgr = ActMgr()
        self.action_mask = action_mask # whether record availability of actions in dc
        self.step_mul = step_mul

    def reset(self, obs, **kwargs):
        super(ZergProdActInt, self).reset(obs, **kwargs)
        self.prodstr_mgr.reset(obs)
        self.build_mgr.reset()
        self.unwrapped().mask_size += sum(self.prodstr_mgr.action_space.nvec)

    @property
    def action_space(self):
        if self.append:
            if isinstance(self.inter.action_space, NoneSpace):
                return NoneSpace()
            assert isinstance(self.inter.action_space, spaces.MultiDiscrete)
            return spaces.MultiDiscrete(list(self.inter.action_space.nvec) +
                                        list(self.prodstr_mgr.action_space.nvec))
        else:
            return self.prodstr_mgr.action_space

    @property
    def inter_action_len(self):
        return len(self.inter.action_space.nvec)

    def act_trans(self, action):
        if self.append:
            act_old = super(ZergProdActInt, self).act_trans(action[0:self.inter_action_len])
            act = action[self.inter_action_len:]
        else:
            super(ZergProdActInt, self).act_trans([])
            act_old = []
            act = action
        if (self.unwrapped().steps - 1) % self.step_mul != 0:
            act = [0]
        self.prodstr_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr, action=act)
        self.build_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        raw_actions = self.am_mgr.pop_actions()
        return act_old + raw_actions

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if self.unwrapped().steps % self.step_mul == 0:
            if self.action_mask:
                action_avail = self.prodstr_mgr.action_avail(self.unwrapped().dc)
            else:
                action_avail = [1] * sum(self.prodstr_mgr.action_space.nvec)
        else:
            action_avail = []
            [action_avail.extend([1] + [0]*(n-1)) for n in self.prodstr_mgr.action_space.nvec]
        self.unwrapped().mask.extend(action_avail)
        return obs