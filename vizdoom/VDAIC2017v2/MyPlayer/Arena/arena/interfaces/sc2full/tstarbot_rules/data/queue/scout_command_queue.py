from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from enum import Enum, unique

from ...data.queue.command_queue_base import CommandQueueBase


@unique
class ScoutCommandType(Enum):
  MOVE = 0


class ScoutCommandQueue(CommandQueueBase):
  def __init__(self):
    SCOUT_CMD_ID_BASE = 300000000
    super(ScoutCommandQueue, self).__init__(SCOUT_CMD_ID_BASE)
