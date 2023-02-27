from tleague.actors.agent import PGAgent2
import tpolicies.net_zoo.mnet_v5.mnet_v5 as policy
import joblib


def PPOAgent2_test():
  from timitate.lib4.pb2all_converter import PB2AllConverter

  policy_config = {
    'test': True,
    'rl': False,
    'use_lstm': True,
    'batch_size': 1,
    'rollout_len': 1,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'lstm_cell_type': 'lstm'
  }

  pb2all = PB2AllConverter(input_map_size=(128, 128),
                           output_map_size=(128, 128),
                           dict_space=True)
  ob_space = pb2all.space.spaces[0]
  ac_space = pb2all.space.spaces[1]

  agt = PGAgent2(policy, ob_space, ac_space, n_v=1, scope_name='model',
                 policy_config=policy_config)

  #ob = ob_space.sample()
  #print(ob)
  ob = None
  agt.reset(ob)

  rnn_state = agt.state
  print(rnn_state)

  ob = ob_space.sample()
  print(ob)
  rnn_state = agt.update_state(ob)
  print(rnn_state)


  ob = ob_space.sample()
  _dummy = agt.forward_squeezed(ob)
  a, v, last_state, neglogp = agt.forward_squeezed(ob)
  print(a)
  print(v)
  print(last_state)
  print(neglogp)

  for i in range(3):
    ob = ob_space.sample()
    aa = agt.step(ob)
    print(aa)

  print('*****************')
  print('testing all done.')
  print('*****************')


def PPOAgent2_save_load_test():
  from timitate.lib4.pb2all_converter import PB2AllConverter

  policy_config = {
    'test': True,
    'rl': False,
    'use_lstm': True,
    'batch_size': 1,
    'rollout_len': 1,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'lstm_cell_type': 'lstm'
  }

  pb2all = PB2AllConverter(input_map_size=(128, 128),
                           output_map_size=(128, 128),
                           dict_space=True)
  ob_space = pb2all.space.spaces[0]
  ac_space = pb2all.space.spaces[1]

  agt = PGAgent2(policy, ob_space, ac_space, n_v=1,
                 scope_name='model_save_load_test',
                 policy_config=policy_config)

  pp = agt.sess.run(agt.params)
  joblib.dump(pp, 'tmp.model')

  loaded_params = joblib.load('tmp.model')
  agt.load_model(loaded_params)


  print('*****************')
  print('testing all done.')
  print('*****************')


if __name__ == '__main__':
  PPOAgent2_test()
  PPOAgent2_save_load_test()
