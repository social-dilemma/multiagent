"""
test.py
Jared Weinstein

Play with a trained agent
"""

import argparse
import curses
from pycolab import human_ui

from environment.example import *

import pygame
import pdb
import time

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

class GameVisualizer(object):
    WIDTH = 20
    HEIGHT = 20
    MARGIN = 5

    COLORS = {
        '#': WHITE,          # Impassable wall
        '0': GREEN,          # Player 1
        '1': BLUE,           # Player 2
        ' ': BLACK,          # Background
        'missing': RED,
        'background': BLACK
    }

    def __init__(self, env, delay, pov='agent-0'):
        self.env = env
        self.pov = pov
        self.delay = delay

        # set up agents
        # set up user input


    def run(self):
        # set up pygame grid environment
        width = self.env.game.cols * (self.WIDTH + self.MARGIN) + self.MARGIN
        height = self.env.game.rows * (self.HEIGHT + self.MARGIN) + self.MARGIN
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode([width, height])

        # reset pycolab game
        observations = self.env.reset()
        grid = observations[self.pov]
        self._render(screen, clock, grid)

        # primary game loop
        while not self.env.game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # get users actions
            # get artificial agents actions
            actions = {'agent-0': 3, 'agent-1': 4}
            observation, reward, dones, infos = self.env.step(actions)

            self._render(screen, clock, observation[self.pov])
            time.sleep(self.delay)

        print("GAME OVER ===========")
        print("REWARD:  {}".format(reward))

        pygame.quit()

    def _render(self, screen, clock, plot):
        screen.fill(self.COLORS['background'])

        for i in range(len(plot)):
            row = plot[i]
            for j in range(len(row)):
                char = chr(row[j])
                self._draw_rect(screen, char, i, j)

        clock.tick(600)
        pygame.display.flip()

    def _draw_rect(self, screen, char, row, column):
        if char in self.COLORS:
            color = self.COLORS[char]
        else:
            print('missing color')
            color = self.COLORS['missing']

        x = (self.MARGIN + self.WIDTH) * column + self.MARGIN
        y = (self.MARGIN + self.HEIGHT) * row + self.MARGIN
        pygame.draw.rect(screen, color, [x, y, self.WIDTH, self.HEIGHT])


def main(args):
    env = ExampleEnvironment()
    game = env.game

    # Load trained models
    # TODO

    visualizer = GameVisualizer(env, .2)
    visualizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Train and play multiagent sequential dilemma')
    args = parser.parse_args()
    main(args)

