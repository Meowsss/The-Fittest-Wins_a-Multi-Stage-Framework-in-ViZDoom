from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import uuid
from multiprocessing import Process

import timeout_decorator
from tleague.actors.base_actor import ActorCommunicator
from tleague.base_league_mgr import ActorTask
from tleague.base_league_mgr import LeagueManagerClient
from tleague.hyperparam_mgr.hyperparam_types import MutableHyperparam
from tleague.league_mgrs.league_mgr import LeagueManager
from tleague.learners.base_learner import LearnerCommunicator
from tleague.model_pool import ModelPool, ModelPoolClient


class TestActorCommunicator(unittest.TestCase):

  def setUp(self):
    self._model_process = Process(
        target=lambda: ModelPool(ports="11001:11006").run())
    self._league_process = Process(
        target=lambda: LeagueManager(
            port="11007", model_pool_addrs=["localhost:11001:11006"],
            mutable_hyperparam_type='MutableHyperparam').run())

    self._model_process.start()
    self._league_process.start()

    model_client = ModelPoolClient(model_pool_addrs=["localhost:11001:11006"])
    model_client.push_model(None, MutableHyperparam(), str(uuid.uuid1()))
    league_client = LeagueManagerClient(league_mgr_addr="localhost:11007")
    league_client.request_learner_task(learner_id=str(uuid.uuid1()))


  def tearDown(self):
    self._model_process.terminate()
    self._league_process.terminate()

  @timeout_decorator.timeout(1)
  def test_pull_task(self):
    actor_comm = ActorCommunicator(league_mgr_addr="localhost:11007",
                                   model_pool_addrs=["localhost:11001:11006"],
                                   learner_addr=None)
    task = actor_comm.pull_task()
    self.assertTrue(isinstance(task, ActorTask))

  @timeout_decorator.timeout(1)
  def test_pull_model(self):
    actor_comm = ActorCommunicator(league_mgr_addr="localhost:11007",
                                   model_pool_addrs=["localhost:11001:11006"],
                                   learner_addr=None)
    model = actor_comm.pull_model(str(uuid.uuid1()))
    self.assertEqual(model, None)

  @timeout_decorator.timeout(1)
  def test_push_result(self):
    actor_comm = ActorCommunicator(league_mgr_addr="localhost:11007",
                                   model_pool_addrs=["localhost:11001:11006"],
                                   learner_addr=None)
    actor_comm.push_result(str(uuid.uuid1()), str(uuid.uuid1()), 1)

  @timeout_decorator.timeout(2)
  def test_push_data(self):
    learner_comm = LearnerCommunicator(
        league_mgr_addr="localhost:11007",
        model_pool_addrs=["localhost:11001:11006"],
        learner_ports="10010:10011")
    actor_comm = ActorCommunicator(league_mgr_addr="localhost:11007",
                                   model_pool_addrs=["localhost:11001:11006"],
                                   learner_addr="localhost:10010:10011")
    data_sent = "any python object"
    actor_comm.push_data(data_sent)
    data_recv = learner_comm.pull_data()
    self.assertEqual(data_recv, data_sent)


if __name__ == '__main__':
  unittest.main()
