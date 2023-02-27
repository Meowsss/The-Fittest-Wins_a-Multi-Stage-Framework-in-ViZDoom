# TODO(peng): obsolete, use zerg_scout_auto_int instead
from arena.interfaces.interface import Interface
from .tstarbot_rules.scout.scout_mgr import ZergScoutMgr
from .tstarbot_rules.act.act_mgr import ActMgr


class ZergScoutInt(Interface):
    def __init__(self, inter, **kwargs):
        super(ZergScoutInt, self).__init__(inter)
        self.am_mgr = ActMgr()
        self.scout_mgr = ZergScoutMgr(self.unwrapped().dc)

    def reset(self, obs, **kwargs):
        super(ZergScoutInt, self).reset(obs, **kwargs)
        self.scout_mgr.reset()

    def act_trans(self, act):
        act = self.inter.act_trans(act)
        self.scout_mgr.update(dc=self.unwrapped().dc, am=self.am_mgr)
        raw_actions = self.am_mgr.pop_actions()
        return act + raw_actions
