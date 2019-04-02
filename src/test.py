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

WIDTH = 20
HEIGHT = 20
MARGIN = 5

class ExampleConfig():
    env = ExampleEnvironment()

    colors = {
        '#': WHITE,          # Impassable wall
        '0': GREEN,          # Player 1
        '1': BLUE,           # Player 2
        ' ': BLACK,          # Background
        'missing': RED,
        'background': BLACK
    }

    action_map = {
        pygame.K_UP: Actions.UP,
        pygame.K_DOWN: Actions.DOWN,
        pygame.K_RIGHT: Actions.RIGHT,
        pygame.K_LEFT: Actions.LEFT,
        pygame.K_s: Actions.STAY
    }


def run(config, delay, interactive, pov='agent-0'):
    # set up pygame grid environment
    env = config.env
    width = env.game.cols * (WIDTH + MARGIN) + MARGIN
    height = env.game.rows * (HEIGHT + MARGIN) + MARGIN
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode([width, height])

    # setup artificial agents

    # reset pycolab game
    observations = env.reset()
    grid = observations[pov]
    _render(screen, grid, config.colors)
    pygame.display.flip()

    # primary game loop
    while not env.game.game_over:
        update = _update_game(env, interactive, config.action_map)
        obs, reward, dones = update
        _render(screen, obs[pov], config.colors)
        pygame.display.flip()
        clock.tick(60)
        time.sleep(delay)

    print("FINAL REWARD:  {}".format(reward))
    pygame.quit()

def _update_game(env, interactive, action_map):
    # handle pygame input
    user_action = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # get users actions
            if event.type == pygame.KEYDOWN:
                user_action = action_map.get(event.key)

        if not interactive or user_action != None:
            break

    # get artificial agents actions
    actions = {'agent-0': Actions.RIGHT, 'agent-1': Actions.LEFT}

    # override action with user input
    if interactive:
        if user_action == None: user_action = Actions.STAY
        actions['agent-0'] = user_action

    # step the environment
    observation, reward, dones, infos = env.step(actions)
    return (observation, reward, dones)

def _render(screen, plot, colors):
    screen.fill(colors['background'])

    for n_row in range(len(plot)):
        row = plot[n_row]
        for n_col in range(len(row)):
            char = chr(row[n_col])
            if char in colors:
                color = colors[char]
            else:
                print('missing color')
                color = colors['missing']

            x = (MARGIN + WIDTH) * n_col + MARGIN
            y = (MARGIN + HEIGHT) * n_row + MARGIN
            pygame.draw.rect(screen, color, [x, y, WIDTH, HEIGHT])

def main(args):
    configs = {
        'example': ExampleConfig()
    }

    config = configs.get(args.env)
    if config == None:
        print('invalid environment')
        return

    # Load trained models
    # TODO

    if args.interactive: args.delay = 0.0
    run(config, args.delay, args.interactive)

def parse_args():
    parser = argparse.ArgumentParser(description=
            'Train and play multiagent sequential dilemma')
    parser.add_argument('--skip_visual', action='store_true',
                    help='skip pygame rendering to increase speed')
    parser.add_argument('--delay', metavar='delay', default=0.6, type=float,
                    help='delay between step')
    parser.add_argument('--env', type=str, default='example',
                    help='Desired enviromnment')
    parser.add_argument('--play', dest='interactive', action='store_true',
                    help='replaces one agent with user input')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
