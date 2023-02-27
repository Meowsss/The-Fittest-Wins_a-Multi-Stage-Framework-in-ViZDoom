#!/usr/bin/env python

from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import vizdoom as vzd
from agent import Runner
import time

game = vzd.DoomGame()
game.load_config("config/my_custom_config.cfg")


parser = ArgumentParser("F1 for ViZDoom Copmetition at CIG 2017.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--color', metavar="F1_COLOR", dest='f1_color',
                        default=0, type=int,
                        help='0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue')
parser.add_argument('-w', '--watch', dest='watch', action='store_const',
                        default=False, const=True,
                        help='window visible')
parser.add_argument('-mode', '--mode', metavar="MODE", dest='mode',
                        default=1, type=int,
                        help='1 for PLAYER, 2 for ASYNC_PLAYER')
parser.add_argument('-port', '--port', metavar="PORT", dest='port',
                        default=0, type=int,
                        help='port')


args = parser.parse_args()
colorset = args.f1_color
watch = args.watch
mode = args.mode
port = args.port

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+colorset {}".format(colorset))
game.add_game_args("+name F1")
if port != 0:
    game.add_game_args("-port {}".format(port))

if watch:
    game.set_window_visible(True)
else:
    game.set_window_visible(False)

if mode == 1:
    game.set_mode(vzd.Mode.PLAYER)
else:
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

game.set_screen_resolution(vzd.ScreenResolution.RES_512X384)

game.init()

print("F1 joined the party!")


runner =  Runner(game)


# Play until the game (episode) is over.
while not game.is_episode_finished():
    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()
	print('F1 is back!')
        # Or observe the game until automatic respawn.
        #game.advance_action();
        #continue;
	
    runner.step()
        
    


game.close()
