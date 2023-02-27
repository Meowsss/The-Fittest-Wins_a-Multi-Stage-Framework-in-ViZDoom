""" Interface wrappers """
from gym import spaces
from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace
from arena.interfaces.sc2full.rules.zerg_simple_prod_rules import ZergSimpleProdMgr
from arena.interfaces.sc2full.rules.zerg_simple_com_rules import ZergSimpleComMgr
from .tstarbot_rules.building.building_mgr import ZergBuildingMgr
from .tstarbot_rules.resource.resource_mgr import ZergResourceMgr
from .tstarbot_rules.act.act_mgr import ActMgr
from .tstarbot_rules.combat.combat_mgr import ZergCombatMgr


class ZergSimpleFlatActInt(Interface):
    def __init__(self, inter, auto_resource=True, auto_larva=True,
                 auto_pre=True, auto_supply=True, keep_order=False,
                 action_mask=False):
        super(ZergSimpleFlatActInt, self).__init__(inter)
        self.append = False
        self.prodstr_mgr = ZergSimpleProdMgr(self.unwrapped().dc, auto_resource=auto_resource,
                                             auto_larva=auto_larva, auto_pre=auto_pre,
                                             auto_supply=auto_supply, keep_order=keep_order)
        self.comstr_mgr = ZergSimpleComMgr(self.unwrapped().dc)

        self.build_mgr = ZergBuildingMgr(self.unwrapped().dc)
        self.resource_mgr = ZergResourceMgr(self.unwrapped().dc)
        self.micro_mgr = ZergCombatMgr(self.unwrapped().dc)

        self.am_mgr = ActMgr()
        self.action_mask = action_mask  # whether record availability of actions in dc

    def reset(self, obs, **kwargs):
        super(ZergSimpleFlatActInt, self).reset(obs, **kwargs)
        self.prodstr_mgr.reset(obs)
        self.comstr_mgr.reset()
        self.build_mgr.reset()
        self.resource_mgr.reset()
        self.micro_mgr.reset()

        if self.action_mask:
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
            return spaces.Discrete(self.prodstr_mgr.action_space.n + 2)

    @property
    def inter_action_len(self):
        return len(self.inter.action_space.nvec)

    def act_trans(self, action):
        if self.append:
            act_old = super(ZergSimpleFlatActInt, self).act_trans(action[0:self.inter_action_len])
            self.prodstr_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr, action=action[self.inter_action_len:])
        else:
            act_old = []
            if action < self.prodstr_mgr.action_space.n:
                self.prodstr_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr, action=action)
            else:
                self.comstr_mgr.update(dc=self.unwrapped().dc, exe_cmd=action-self.prodstr_mgr.action_space.n)

        self.build_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        self.resource_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        self.micro_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        raw_actions = self.am_mgr.pop_actions()
        return act_old + raw_actions

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if self.action_mask:
            action_avail = self.prodstr_mgr.action_avail(self.unwrapped().dc)
            self.unwrapped().mask.extend(action_avail)
        return obs
