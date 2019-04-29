"""
test.py
Jared Weinstein

Renders an instance of an environment and trained agent
Pygame renders frontend of the game and handles user input
"""

import pygame
import time
import json
import argparse
import curses
from pycolab import human_ui
from pathlib import Path

import ray
from ray.rllib.models import ModelCatalog
from ray.cloudpickle import cloudpickle
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from models.conv_to_fc_net import ConvToFCNet
from config import *

def _get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path / 'params.json' # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())

    pklfile = path / 'params.pkl'  # params.pkl is the config file
    with open(pklfile, 'rb') as file:
        pkldata = cloudpickle.load(file)

    return jsondata, pkldata

def _color_map(colors, char_mapping):
    # colors mapping
    for key, value in char_mapping:
        val = colors.get(chr(key))
        del colors[chr(key)]
        colors[value] = val
    return colors

"""
Visualizer

Renders a game
"""
class Visualizer():
    def __init__(self, args):
        ray.init(num_cpus=1)
        result_dir = Path(args.result_dir)

        # register our environment
        setup, config = self._register_env(result_dir)

        # initialize agents
        agent, agent_info = self._load_agents(result_dir, config)
        env = agent.local_evaluator.env

        # color mapping
        char_map = env.mapping.items()
        colors = _color_map(setup.colors, char_map)

        # Run on only one cpu for rendering purposes if possible; A3C requires two (?)
        config_run = config['env_config']['run']
        # if config_run == 'A3C':
        #     config['num_workers'] = 1
        config['num_workers'] = 0

        # override inconsistent argument
        if args.skip_visual:
            args.delay = 0.0
            args.interactive = False
        if args.interactive: args.delay = 0.0

        # set necessary globals
        self.visualize = not args.skip_visual
        self.delay = args.delay
        self.interactive = args.interactive
        self.pov = args.pov
        self.setup = setup
        self.colors = colors
        self.agent = agent
        self.agent_info = agent_info
        self.action_map = setup.action_map

    def run(self):
        env = self.agent.local_evaluator.env
        state = env.reset()

        # setup pygame
        if self.visualize:
            width = env.game.cols * (WIDTH + MARGIN) + MARGIN
            height = env.game.rows * (HEIGHT + MARGIN) + MARGIN + 20
            pygame.init()
            clock = pygame.time.Clock()
            screen = pygame.display.set_mode([width, height])
            # render first frame
            grid = state[self.pov]
            self._render(screen, grid, None, self.colors)
            pygame.display.flip()

        # primary game loop
        while not env.game.game_over:
            update = self._update_game(env, state)
            if update == None: break
            state, reward, dones = update
            del dones

            if self.visualize:
                self._render(screen, state[self.pov], reward, self.colors)
                pygame.display.flip()
                clock.tick(60)
                time.sleep(self.delay)

        print("FINAL REWARD:  {}".format(reward))
        pygame.quit()

    """
    MARK: helper methods for core game loop
    """
    def _agent_actions(self, state):
        policy_agent_mapping = self.agent_info['policy_map']
        use_lstm = self.agent_info['use_lstm']
        state_init = self.agent_info['init_state'] # TODO: does this need to get updated directly
        agent = self.agent

        action_dict = {}
        mapping_cache = {}
        for agent_id in state.keys():
            a_state = state[agent_id]
            if a_state is not None:
                policy_id = mapping_cache.setdefault(
                    agent_id, policy_agent_mapping(agent_id))
                p_use_lstm = use_lstm[policy_id]
                if p_use_lstm:
                    a_action, p_state_init, _ = agent.compute_action(
                        a_state,
                        state=state_init[policy_id],
                        policy_id=policy_id)
                    state_init[policy_id] = p_state_init
                else:
                    a_action = agent.compute_action(
                        a_state, policy_id=policy_id)
                action_dict[agent_id] = a_action
        return action_dict

    def _update_game(self, env, state):
        # handle pygame input
        user_action = None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                # get users actions
                if event.type == pygame.KEYDOWN:
                    user_action = self.action_map.get(event.key)

            if not self.interactive or user_action != None:
                break

        actions = self._agent_actions(state)

        # override action with user input
        if self.interactive:
            actions[self.pov] = user_action

        # step the environment
        state, rewards, dones, infos = env.step(actions)

        print("ACTIONS {}".format(actions))
        print("REWARDS {}\r\r".format(rewards))

        return (state, rewards, dones)

    def _render(self, screen, plot, reward, colors):
        screen.fill(colors['background'])

        for n_row in range(len(plot)):
            row = plot[n_row]
            for n_col in range(len(row)):
                char = (row[n_col])[0]
                if char in colors:
                    color = colors[char]
                else:
                    print(char)
                    print('missing color')
                    color = colors['missing']

                x = (MARGIN + WIDTH) * n_col + MARGIN
                y = (MARGIN + HEIGHT) * n_row + MARGIN
                pygame.draw.rect(screen, color, [x, y, WIDTH, HEIGHT])

        # TODO: display reward on screen

    """
    MARK: methods to help with setup
    """
    def _register_env(self, result_dir):
        config, pkl = _get_rllib_config(result_dir)

        config['multiagent'] = pkl['multiagent']
        if not config.get('multiagent', {}).get('policy_graphs', {}):
            print('something went wrong with multiagent policy graph')
            import pdb; pdb.set_trace()

        # Create and register a gym+rllib env
        env_creator = pkl['env_config']['func_create']
        env_name = config['env_config']['env_name']
        register_env(env_name, env_creator.func)
        if env_name == 'example_env':
            setup = ExampleSetup()
        elif env_name == 'prison_env':
            setup = PrisonSetup()
        else:
            print('invalid environment')
            setup = None

        return setup, config

    def _load_agents(self, result_dir, config):
        env_name = config['env_config']['env_name']
        agent_dict = {}

        # Determine agent and checkpoint
        ModelCatalog.register_custom_model("conv_to_fc_net", ConvToFCNet)
        config_run = config['env_config']['run']
        agent_cls = get_agent_class(config_run)
        print(
            """ WARNING: the script is about to instantiate the A3C agents
                  however this has undersired asynchronous behavior as they
                  simultaneously interact with the environment. Please wait until
                  all interactions have finished before continuing. Apologies
                  this is an error that needs to be fixed
            """)
        import pdb.set_trace()
        agent = agent_cls(env=env_name, config=config)
        import pdb; pdb.set_trace()
        # TODO: My hacky solution is to wait till the print lines
        #       have finished and then manually continue.
        #       This needs to be fixed.

        # create the agent that will be used to compute the actions
        checkpoint = result_dir / ('checkpoint_' + args.checkpoint_num)
        checkpoint = checkpoint / ('checkpoint-' + args.checkpoint_num)
        print('Loading checkpoint', checkpoint)
        agent.restore(str(checkpoint))
        policy_agent_map = agent.config["multiagent"][
                    "policy_mapping_fn"]
        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

        agent_dict['use_lstm'] = use_lstm
        agent_dict['policy_map'] = policy_agent_map
        agent_dict['init_state'] = state_init
        return agent, agent_dict


def parse_args():
    parser = argparse.ArgumentParser(description=
            'Train and play multiagent sequential dilemma')

    # mandatory args
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional args
    parser.add_argument('--skip_visual', action='store_true',
                    help='skip pygame rendering to increase speed')
    parser.add_argument('--delay', metavar='delay', default=0.6, type=float,
                    help='delay between step')
    parser.add_argument('--play', dest='interactive', action='store_true',
                    help='replaces one agent with user input')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.pov = 'agent-0' # todo: add argument for this
    vis = Visualizer(args)
    vis.run()
