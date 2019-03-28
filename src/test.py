"""
test.py
Jared Weinstein

Play with a trained agent

Pycolab's human_ui.py provided the base for most of
this code. Modifications have been made primarily to
visualize game play of trained artificial agents.
"""

import argparse
import curses
from pycolab import human_ui

from environment.example import *

class GameVisualizer(object):
    def __init__(self):




def main(args):
    # Create the Game
    agents = []
    for i in range(2):
        name = 'agent-' + str(i)
        agent = ExampleAgent(name, i, str(i))
        agents.append(agent)

    env = ExampleEnvironment(agents)
    game = env.game

    # Load trained models
    # TODO

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
            'Train and play multiagent sequential dilemma')
    args = parser.parse_args()
    main(args)

