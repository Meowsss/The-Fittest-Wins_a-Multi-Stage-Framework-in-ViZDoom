""" Auto Action Interface for Zerg"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from arena.interfaces.interface import Interface
from .tstarbot_rules.production_strategy.production_mgr import ZergProductionMgr as ZergProductStrategyMgr
from .tstarbot_rules.building.building_mgr import ZergBuildingMgr
from .tstarbot_rules.resource.resource_mgr import ZergResourceMgr
from .tstarbot_rules.combat_strategy.combat_strategy_mgr import ZergStrategyMgr as ZergCombatStrategyMgr
from .tstarbot_rules.combat.combat_mgr import ZergCombatMgr as ZergCombatMicroMgr
from .tstarbot_rules.scout.scout_mgr import ZergScoutMgr


class TstarbotAutoInt(Interface):
  """ Wraps TStarBot XXXMgr and takes action based on the pre-defined rules """

  def __init__(self, inter, mgr_cls):
    super(TstarbotAutoInt, self).__init__(inter)
    self._mgr = mgr_cls(dc=self.unwrapped().dc)

  def reset(self, obs, **kwargs):
    super(TstarbotAutoInt, self).reset(obs, **kwargs)
    self._mgr.reset()

  def act_trans(self, action):
    rest_act = self.inter.act_trans(action)
    rest_act = rest_act if type(rest_act) is list else [rest_act]
    # determine me action (automatically, do not rely on the action passed in)
    self._mgr.update(dc=self.unwrapped().dc, am=self.unwrapped().am)
    me_act = self.unwrapped().am.pop_actions()
    return me_act + rest_act


class ZergProductStrategyAutoInt(TstarbotAutoInt):
  """ Wraps TStarBot ZergProductStrategyMgr """
  def __init__(self, inter):
    super(ZergProductStrategyAutoInt, self).__init__(inter, mgr_cls=ZergProductStrategyMgr)


class ZergBuildingAutoInt(TstarbotAutoInt):
  """ Wraps TStarBot ZergBuildingMgr """
  def __init__(self, inter):
    super(ZergBuildingAutoInt, self).__init__(inter, mgr_cls=ZergBuildingMgr)


class ZergResourceAutoInt(TstarbotAutoInt):
  """ Wraps TStarBot ZergResourceMgr """
  def __init__(self, inter):
    super(ZergResourceAutoInt, self).__init__(inter, mgr_cls=ZergResourceMgr)


class ZergCombatStrategyAutoInt(TstarbotAutoInt):
  """ Wraps TStarBot ZergCombatStrategyMgr """
  def __init__(self, inter):
    super(ZergCombatStrategyAutoInt, self).__init__(inter, mgr_cls=ZergCombatStrategyMgr)


class ZergCombatMicroAutoInt(TstarbotAutoInt):
  """ Wraps TStarBot ZergCombatMicroMgr """
  def __init__(self, inter):
    super(ZergCombatMicroAutoInt, self).__init__(inter, mgr_cls=ZergCombatMicroMgr)


class ZergScoutAutoInt(TstarbotAutoInt):
  """ Wraps TStarBot ZergCombatMicroMgr """
  def __init__(self, inter):
    super(ZergScoutAutoInt, self).__init__(inter, mgr_cls=ZergScoutMgr)

