from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BasePolicy:

  def step(self, ob):
    pass

  def value(self, ob):
    pass

  def act(self, ob):
    pass
