from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from enum import Enum
from enum import unique

from .pool_base import PoolBase


@unique
class OppoOpeningTactics(Enum):
  UNKNOWN = 0
  BANELING_RUSH = 1
  ROACH_RUSH = 2
  ZERGROACH_RUSH = 3
  ROACH_SUPPRESS = 4


class OppoPool(PoolBase):
  def __init__(self):
    super(OppoPool, self).__init__()
    self.opening_tactics = None

  def reset(self):
    self.opening_tactics = None
