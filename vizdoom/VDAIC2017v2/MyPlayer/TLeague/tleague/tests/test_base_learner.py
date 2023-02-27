from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import uuid
from multiprocessing import Process

import timeout_decorator
from tleague.actors.base_actor import ActorCommunicator
from tleague.base_league_mgr import LearnerTask
from tleague.league_mgrs.league_mgr import LeagueManager
from tleague.learners.base_learner import LearnerCommunicator
from tleague.model_pool import ModelPool


class TestLearnerCommunicator(unittest.TestCase):

  def setUp(self):
    self._model_process = Process(
        target=lambda: ModelPool(ports="11001:11006").run())
    self._league_process = Process(
        target=lambda: LeagueManager(
            port="11007", model_pool_addrs=["localhost:11001:11006"],
            mutable_hyperparam_type='MutableHyperparam').run())

    self._model_process.start()
    self._league_process.start()

  def tearDown(self):
    self._model_process.terminate()
    self._league_process.terminate()

  @timeout_decorator.timeout(1)
  def test_pull_task(self):
    learner_comm = LearnerCommunicator(
        league_mgr_addr="localhost:11007",
        model_pool_addrs=["localhost:11001:11006"],
        learner_ports="11010:11011")
    task = learner_comm.pull_task()
    self.assertTrue(isinstance(task, LearnerTask))

  @timeout_decorator.timeout(1)
  def test_pull_model(self):
    learner_comm = LearnerCommunicator(
        league_mgr_addr="localhost:11007",
        model_pool_addrs=["localhost:11001:11006"],
        learner_ports="11020:11021")
    model = learner_comm.pull_model(str(uuid.uuid1()))
    self.assertEqual(model, None)

  @timeout_decorator.timeout(1)
  def test_push_model(self):
    learner_comm = LearnerCommunicator(
        league_mgr_addr="localhost:11007",
        model_pool_addrs=["localhost:11001:11006"],
        learner_ports="11030:11031")
    learner_comm.push_model(
        model=None, hyperparam=None, model_key=str(uuid.uuid1()))

  @timeout_decorator.timeout(2)
  def test_pull_data(self):
    learner_comm = LearnerCommunicator(
        league_mgr_addr="localhost:11007",
        model_pool_addrs=["localhost:11001:11006"],
        learner_ports="10040:10041")
    actor_comm = ActorCommunicator(league_mgr_addr="localhost:11007",
                                   model_pool_addrs=["localhost:11001:11006"],
                                   learner_addr="localhost:10040:10041")
    data_sent = "any python object"
    actor_comm.push_data(data_sent)
    data_recv = learner_comm.pull_data()
    self.assertEqual(data_recv, data_sent)


if __name__ == '__main__':
  unittest.main()
