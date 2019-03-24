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
from ray.rllib.env import MultiAgentEnv

from gym.spaces import Box, Discrete
from environment.base_class import RewardSprite, Agent

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
    def __init__(self, agents):
        if len(agents) != 2:
            raise Exception('The example environment must be inititalized with 2 agents')

        self.agents = agents
        self.game = self._make_game()

    def _make_game(self):
        """Builds and returns a sample environment"""

        sprites = {}
        for agent in self.agents:
            sprites[agent.char] = ascii_art.Partial(self.ExampleSprite, agent.index, len(self.agents))

        return ascii_art.ascii_art_to_game(
                GAME_ART,
                what_lies_beneath=' ',
                sprites = sprites
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

        # step the environment using given actions
        actions = [0, 1]
        observation, reward, discount = self.game.play(actions)
        del discount

        observations = {}
        dones = {}
        rewards = {}
        info = {}

        for agent in self.agents:
            view = observation.board
            done = False
            reward = reward

            # print(view)
            # print(done)
            # print(reward)

            observations[agent.name] = view
            dones[agent.name] = done
            rewards[agent.name] = 0 #TODO
            info[agent.name] = None


        dones['__all__'] = self.game.game_over

        return observations, rewards, dones, info


    def reset(self):
        """
        Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """

        self.game = self._make_game()
        observation, reward, discount = self.game.its_showtime()
        del reward, discount

        # TODO: implement partial observability. Every agent currently
        #       gets the entire board
        observations = {}
        for agent in self.agents:
            observations[agent.name] = observation.board
        return observations

    class ExampleSprite(RewardSprite):
        """ Sprite representing player for Pycolab """

        def __init__(self, corner, position, character, index, n_unique):
            super(ExampleEnvironment.ExampleSprite, self).__init__(corner, position, character, index, n_unique)

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


class ExampleAgent(Agent):
    """
    Agent Representation for RLLib
    """
    def __init__(self, name, index, char):
        super(ExampleAgent, self).__init__(name, index, char)

    @property
    def action_space(self):
        return Discrete(len(Actions))

    @property
    def observation_space(self):
        # TODO: Verify accuracy
        #       might be 22, 2
        return Box(
                low=-1,
                high=1,
                shape=(2, 22),
                dtype=np.float32)

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
