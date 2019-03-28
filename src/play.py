import sys
import curses
from pycolab import human_ui

from environment.example import *
import argparse

"""
for a human to play a specific game
"""

COLORS = {
            '#': (999, 999, 999),   # Impassable wall
            '0': (500, 100, 0),     # Player 1
            '1': (0, 500, 100),     # Player 2
         }

env = ExampleEnvironment()
game = env.game

ui = human_ui.CursesUi(
        delay=500,
        colour_fg=COLORS,
        keys_to_actions={
            curses.KEY_LEFT: [Actions.STAY, Actions.LEFT],
            curses.KEY_RIGHT: [Actions.STAY, Actions.RIGHT],
            'a': [Actions.LEFT, Actions.STAY],
            'd': [Actions.RIGHT, Actions.STAY],
            -1: [Actions.STAY, Actions.STAY]
        })
ui.play(game)
sys.exit()
