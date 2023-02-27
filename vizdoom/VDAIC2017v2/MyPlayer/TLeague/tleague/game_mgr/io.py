from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
from os import path

import numpy as np

from . import game_mgrs

### This file is no longer used, keep it as compatibility for old version of SC2AgentZoo

LEAGUE_MGR_FILENAME = 'league_mgr.payoff'


def save_game_mgr(checkpoint_dir, game_mgr):
  filepath = os.path.join(checkpoint_dir, LEAGUE_MGR_FILENAME)
  with open(filepath, 'wb') as f:
    pickle.dump(game_mgr.players, f)
    pickle.dump(game_mgr.finished_matches, f)
    pickle.dump(game_mgr.finished_match_counter, f)
    pickle.dump(game_mgr.sum_outcome, f)


def load_game_mgr(checkpoint_dir, game_mgr_type='RefCountGameMgr'):
  """ Load game manager from checkpoint_dir

  Caller must assure the game_mgr_type is consistent with the one stored in
  checkpoint_dir """
  filepath = os.path.join(checkpoint_dir, LEAGUE_MGR_FILENAME)
  with open(filepath, 'rb') as f:
    players = pickle.load(f)
    finished_matches = pickle.load(f)
    finished_match_counter = pickle.load(f)
    sum_outcome = pickle.load(f)

  model_keys = set()
  with open(path.join(checkpoint_dir, 'filename.list'), 'rt') as f:
    for model_fn in f:
      if model_fn.strip().endswith('.model'):
        # an eligible model_fn line looks like xxxx:yyyy_timestamp.model
        model_keys.add(model_fn.strip().split('.')[0][:-15])

  missing_idx = [i for i, p in enumerate(players) if p not in model_keys]
  if len(missing_idx) > 0:
    print('Players {} is missing in model pool'.format
      ([players[i] for i in missing_idx]))
  finished_match_counter = np.delete(finished_match_counter, missing_idx, 0)
  finished_match_counter = np.delete(finished_match_counter, missing_idx, 1)
  sum_outcome = np.delete(sum_outcome, missing_idx, 0)
  sum_outcome = np.delete(sum_outcome, missing_idx, 1)
  game_mgr_cls = getattr(game_mgrs, game_mgr_type)
  # TODO(pengsun): use kwargs in caller
  game_mgr = game_mgr_cls(pgn_file='example.pgn')
  game_mgr.players = [p for p in players if p in model_keys]
  if game_mgr._use_player2order:
    game_mgr.player2order = dict \
      ([(p, i) for i, p in enumerate(game_mgr.players)])
  game_mgr.finished_matches = {(rp, cp): c for (rp, cp), c in finished_matches.items()
                                    if rp in model_keys and cp in model_keys}
  game_mgr.finished_match_counter = finished_match_counter
  game_mgr.sum_outcome = sum_outcome
  game_mgr.all_match_counter = finished_match_counter

  return game_mgr
