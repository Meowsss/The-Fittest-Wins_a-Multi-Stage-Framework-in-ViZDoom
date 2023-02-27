""" Interfaces for tstarbot data stuff """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib

from arena.interfaces.interface import Interface
from .tstarbot_rules.data.data_context import DataContext
from .tstarbot_rules.act.act_mgr import ActMgr


class ZergDataInt(Interface):
    """ Install TStarBot Data Context & Action Manager """

    def __init__(self, inter, **kwargs):
        super(ZergDataInt, self).__init__(inter)
        config_path = None
        self.config = None
        if kwargs.get('config_path'):  # use the config file
            config_path = kwargs['config_path']
        if config_path:
            self.config = importlib.import_module(config_path)
        self.unwrapped().dc = DataContext(self.config)
        self.unwrapped().am = ActMgr()

    def reset(self, obs, **kwargs):
        super(ZergDataInt, self).reset(obs, **kwargs)
        self.unwrapped().dc.reset()
        self.unwrapped().dc.update(self.unwrapped()._obs)
        self.unwrapped().am.pop_actions()  # clear action buffer
        self.unwrapped().mask = []
        self.unwrapped().mask_size = 0

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        self.unwrapped().dc.update(self.unwrapped()._obs)
        self.unwrapped().mask = []
        return obs
