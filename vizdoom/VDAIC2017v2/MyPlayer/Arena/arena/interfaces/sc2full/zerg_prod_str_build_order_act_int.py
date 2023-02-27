""" Interface wrappers """
from arena.interfaces.interface import Interface
from arena.interfaces.sc2full.rules.zerg_prod_str_build_order_act import ZergProdStrBuildOrderActMgr


class ZergProdStrBuildOrderActInt(Interface):
  def __init__(self, inter, build_order_strategies, act_freq_game_loop=16*60*4):
    super(ZergProdStrBuildOrderActInt, self).__init__(inter)
    self._mgr = ZergProdStrBuildOrderActMgr(dc=self.unwrapped().dc,
                                            build_order_strategies=build_order_strategies,
                                            act_freq_game_loop=act_freq_game_loop)

  def reset(self, obs, **kwargs):
    super(ZergProdStrBuildOrderActInt, self).reset(obs, **kwargs)
    self._mgr.reset_by_obs(obs)

  @property
  def action_space(self):
    return self._mgr.action_space

  @property
  def inter_action_len(self):
    return len(self.inter.action_space.nvec)

  def act_trans(self, action):
    # overwrite mode: abandon the action passed in
    rest_act = self.inter.act_trans([])
    # use the action passed in to determine me action
    self._mgr.update_by_action(dc=self.unwrapped().dc, am=self.unwrapped().am, action=action)
    me_act = self.unwrapped().am.pop_actions()
    return rest_act + me_act

  def obs_trans(self, obs):
    obs = self.inter.obs_trans(obs)
    return obs
