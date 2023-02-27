#!/bin/bash

role=$1
# common args
env=vizdoom_cig2017_track1 && \
env_config="{ \
  'num_players': 1, \
  'num_bots': 0, \
  'is_window_visible':True, \
  'train_mode': 1, \
  'evaluation': 1 \
}" && \
policy=tpolicies.net_zoo.conv_lstm.conv_lstm;
policy_config="{ \
  'test': False, \
  'use_loss_type': 'none', \
  'use_value_head': True, \
  'rollout_len': 1, \
  'use_lstm': True, \
  'nlstm': 128, \
  'hs_len': 256, \
  'lstm_dropout_rate': 0.2, \
  'lstm_layer_norm': True, \
  'weight_decay': 0.00002, \
  'sync_statistics': None, \
}" && \

model='MyPlayer/model/0041_20200921113601.model'

echo "Running as ${role}"
if [ $role == evaluation ]
then
python3 -m tleague.sandbox.run_local_battle_vd \
  --env="${env}" \
  --env_config="${env_config}" \
  --policy1="${policy}" \
  --policy_config="${policy_config}" \
  --model1="${model}" \
  --n_v=1 \
  --episodes=1
fi
