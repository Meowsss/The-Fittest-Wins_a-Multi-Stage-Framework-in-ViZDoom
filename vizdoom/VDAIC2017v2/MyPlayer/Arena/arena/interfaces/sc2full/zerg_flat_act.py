""" Gym env wrappers """
from gym import spaces
from arena.interfaces.interface import Interface


class FlatActInt(Interface):
    def __init__(self, interface, action_mask=False, auto_resource=True, n_area=16):
        super(FlatActInt, self).__init__(interface)
        self.n_area = n_area
        self.action_mask=action_mask
        self.auto_resource=auto_resource
        assert self.inter

    def reset(self, obs, **kwargs):
        super(FlatActInt, self).reset(obs, **kwargs)

    @property
    def action_space(self):
        if self.auto_resource:
            return spaces.Discrete(46+self.n_area)
        else:
            return spaces.Discrete(48+self.n_area)
            
    def act_trans(self, action):
        if not self.auto_resource:
            if action < 46:
                act = [action, 0, 0, 1]
            elif action < 48:
                act = [0, action - 45, 0, 1]
            else:
                act = [0, 0, action - 48, 0]
        else:
            if action < 46:
                act = [action, 0, 1]
            else:
                act = [0, action - 46, 0]
        act = self.inter.act_trans(act)
        return act

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if self.action_mask:
            if not self.auto_resource:
                self.unwrapped().mask = self.unwrapped().mask[0:(48+self.n_area)]
            else:
                self.unwrapped().mask = self.unwrapped().mask[0:(46+self.n_area)]
        return obs
