"""
Example.py
Jared Weinstein

An incredibly simple demonstration of an Pycolab environment
that contains multiple agents receiving unique actions and
rewards.

Major Change: Actions and Rewards become indexed lists
              game.play( [ LEFT, RIGHT ] )

              Every agent with a matching INDEX value receives
              the corresponding action. INDEX is determined by
              the value passed upon Partial(...) initialization.

              No safety checks are done to ensure that action and
              reward lists have the correct length. Please
              behave responsibly.
"""

import curses
import sys
import enum
import argparse

import numpy as np

from pycolab import ascii_art, human_ui
from pycolab.prefab_parts import sprites
from ray.rllib.env import MultiAgentEnv

GAME_ART = ['#   0                #',
            '#             1      #']

class Actions(enum.IntEnum):
    """ Actions for the player """
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class ExampleEnvironment(MultiAgentEnv):
    def __init__(self, num_agents=1):
        self.num_agents = num_agents
        self.game = self._make_game()

    def _make_game(self):
        """Builds and returns a sample environment"""
        return ascii_art.ascii_art_to_game(
                GAME_ART,
                what_lies_beneath=' ',
                sprites= {
                    '0': ascii_art.Partial(PlayerSprite, 0, 2),
                    '1': ascii_art.Partial(PlayerSprite, 1, 2) }
                )

    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update

        Parameters
        ----------
        actions: dict {agent-id: int}
            dict of actions, keyed by agent-id that are passed to the agent. The agent
            interprets the int and converts it to a command

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """

        import pdb; pdb.set_trace()

        # self.beam_pos = []
        # agent_actions = {}
        # for agent_id, action in actions.items():
        #     agent_action = self.agents[agent_id].action_map(action)
        #     agent_actions[agent_id] = agent_action

        # # move
        # self.update_moves(agent_actions)

        # for agent in self.agents.values():
        #     pos = agent.get_pos()
        #     new_char = agent.consume(self.world_map[pos[0], pos[1]])
        #     self.world_map[pos[0], pos[1]] = new_char

        # # execute custom moves like firing
        # self.update_custom_moves(agent_actions)

        # # execute spawning events
        # self.custom_map_update()

        # map_with_agents = self.get_map_with_agents()

        # observations = {}
        # rewards = {}
        # dones = {}
        # info = {}
        # for agent in self.agents.values():
        #     agent.grid = map_with_agents
        #     rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
        #     observations[agent.agent_id] = rgb_arr
        #     rewards[agent.agent_id] = agent.compute_reward()
        #     dones[agent.agent_id] = agent.get_done()
        # dones["__all__"] = np.any(list(dones.values()))
        # return observations, rewards, dones, info

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """

        self.game = self._make_game()
        observation, reward, discount = game.its_showtime()
        del reward, discount

        # TODO: implement partial observability. Every agent currently
        #       gets the entire board
        # TODO: Cleanup dictionary iteration
        observations = {}
        observations = [observation] * self.num_agents

        import pdb; pdb.set_trace()

        # self.beam_pos = []
        # self.agents = {}
        # self.setup_agents()
        # self.reset_map()
        # self.custom_map_update()

        # map_with_agents = self.get_map_with_agents()

        # observations = {}
        # for agent in self.agents.values():
        #     agent.grid = map_with_agents
        #     # agent.grid = util.return_view(map_with_agents, agent.pos,
        #     #                               agent.row_size, agent.col_size)
        #     rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
        #     observations[agent.agent_id] = rgb_arr
        # return observations


class RewardAgent(sprites.MazeWalker):
    """ Sprite representing any player with unique actions and rewards """

    def __init__(self, corner, position, character, index, n_unique):
        self.index = index
        self.n_unique = n_unique

        super(RewardAgent, self).__init__(corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        raise NotImplementedError

    def reward(self, plot, value):
        reward = np.zeros(self.n_unique)
        reward[self.index] = value
        plot.add_reward(reward)


class PlayerSprite(RewardAgent):
    """ Sprite representing the player """

    def __init__(self, corner, position, character, index, n_unique):
        super(PlayerSprite, self).__init__(corner, position, character, index, n_unique)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things # unused

        if actions == None:
            action = Actions.STAY
        else:
            action = actions[self.index]
        del actions

        # action update
        if action == Actions.LEFT:
            self._west(board, the_plot)
        elif action == Actions.RIGHT:
            self._east(board, the_plot)

        # distribute reward
        if self.position[1] == 1:
            self.reward(the_plot, 1)
            the_plot.terminate_episode()
        elif self.position[1] == (self.corner[1] - 2):
            self.reward(the_plot, 100)
            the_plot.terminate_episode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic env to demonstrate pycolab to RLlib connection.')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    del sys.argv

    env = ExampleEnvironment(2)
    game = env.game

    if args.live:
        ui = human_ui.CursesUi(keys_to_actions=
                {
                    curses.KEY_LEFT: [Actions.LEFT, Actions.STAY],
                    curses.KEY_RIGHT: [Actions.STAY, Actions.RIGHT],
                    -1: [Actions.STAY, Actions.STAY]
                },
                delay=200)
        ui.play(game)
        sys.exit()

    if not args.live:
        game.its_showtime()
        while not game.game_over:
            board, reward, discount = game.play([Actions.LEFT, Actions.STAY])
