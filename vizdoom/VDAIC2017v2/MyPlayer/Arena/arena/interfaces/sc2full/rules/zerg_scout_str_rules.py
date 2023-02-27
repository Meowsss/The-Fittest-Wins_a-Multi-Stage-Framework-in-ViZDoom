import pysc2.lib.typeenums as tp
from s2clientprotocol import sc2api_pb2 as sc_pb

from ..tstarbot_rules.scout.tasks import scout_task as st
from ..tstarbot_rules.scout.tasks.explor_task import ScoutExploreTask
from ..tstarbot_rules.data.pool import macro_def as md

class ZergScoutMicroMgr(object):
    def __init__(self, explore_version=st.EXPLORE_V3):
        self._explore_ver = explore_version
        self._tasks = []

    def reset(self):
        self._tasks = []

    def update(self, dc, am, target):
        if target is None:
            am.push_actions(self._noop())
            return
        print('target=', target)

        if not self._dispatch_task(dc, target):
            am.push_actions(self._noop())
            return
        self._check_task(dc)

        actions = []
        # observe the enemy
        units = dc.sd.obs['units']
        view_enemys = []
        for u in units:
            if u.int_attr.alliance == md.AllianceType.ENEMY.value:
                view_enemys.append(u)

        for task in self._tasks:
            act = task.do_task(view_enemys, dc)
            if act is not None:
                actions.append(act)

        if len(actions) > 0:
            am.push_actions(actions)
        return

    def _check_task(self, dc):
        keep_tasks = []
        done_tasks = []
        for task in self._tasks:
            if task.status() == md.ScoutTaskStatus.DONE:
                done_tasks.append(task)
            elif task.status() == md.ScoutTaskStatus.SCOUT_DESTROY:
                done_tasks.append(task)
            else:
                keep_tasks.append(task)

        for task in done_tasks:
            task.post_process()

        self._tasks = keep_tasks

    def _dispatch_task(self, dc, target):
        return self._dispatch_explore_task(dc, target)

    def _dispatch_explore_task(self, dc, target):
        sp = dc.dd.scout_pool
        scout = sp.select_scout()
        if (scout is None):
            return False
        task = ScoutExploreTask(scout, target, sp.home_pos, self._explore_ver)
        scout.is_doing_task = True
        target.has_scout = True
        self._tasks.append(task)
        return True

    def _noop(self):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = tp.ABILITY_ID.INVALID.value
        return [action]


