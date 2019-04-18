import sys
import curses
from pycolab import human_ui

from environment.example import *
from environment.prison import *
import argparse

"""
Quickly play a specific pygame environment using
pygame's curses_ui visualizer
"""


def play_example():
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

def play_prison():
    COLORS = {
                '#': (999, 999, 999),   # Impassable wall
                '0': (500, 100, 0),     # Player 1
                '1': (0, 500, 100),     # Player 2
             }

    env = PrisonEnvironment()
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


if __name__ == "__main__":
    play_prison()
