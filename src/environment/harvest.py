"""
Harvest.py
Jared Weinstein

A shared resource game requiring cooperation
"""

import enum
import numpy as np

from pycolab import ascii_art
from ray.rllib.env import MultiAgentEnv

from gym.spaces import Box, Discrete
from .util.base_class import RewardSprite, Agent

GAME_ART = ['##########################################',
            '#                                        #',
            '#                   0                    #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                                        #',
            '#                   1                    #',
            '#                                        #',
            '##########################################']

class Actions(enum.IntEnum):
    """ Actions for the player """
    STAY = 0
    PUNISH = 1
    RIGHT = 2
    LEFT = 3

class HarvestEnvironment(MultiAgentEnv):
    def __init__(self):
        # create agents
        self.agents = []
        for i in range(2):
            name = 'agent-' + str(i)
            agent = HarvestAgent(name, i, str(i))
            self.agents.append(agent)

        self.game = self._make_game()

    def _make_game(self):
        """Builds and returns a sample environment"""

        sprites = {}
        for agent in self.agents:
            sprites[agent.char] = ascii_art.Partial(self.HarvestSprite, agent.index, len(self.agents))

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
        actions = list(map(lambda a: actions[a.name], self.agents))
        step_observations, step_rewards, step_discount = self.game.play(actions)
        del step_discount

        if step_rewards is None:
           step_rewards = [0] * len(self.agents)

        observations = {}
        dones = {}
        rewards = {}
        info = {}

        for i in range(len(self.agents)):
            observation = step_observations.board.tolist()
            done = self.game.game_over
            reward = step_rewards[i]

            agent = self.agents[i]
            observations[agent.name] = observation
            dones[agent.name] = done
            rewards[agent.name] = reward

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
            observations[agent.name] = observation.board.tolist()
        return observations

    class HarvestSprite(RewardSprite):
        """ Sprite representing player for Pycolab """

        def __init__(self, corner, position, character, index, n_unique):
            super(HarvestEnvironment.HarvestSprite, self).__init__(corner, position, character, index, n_unique)

        def update(self, actions, board, layers, backdrop, things, the_plot):
            action = self.my_action(actions)
            # del layers, backdrop, things, actions

            # action update
            if action == Actions.LEFT:
                self._west(board, the_plot)
            elif action == Actions.RIGHT:
                self._east(board, the_plot)


            # if self.position[1] == 1:
            #     self.reward(the_plot, 1)
            elif self.position[1] == (self.corner[1] - 2):
                self.reward(the_plot, 100)
                the_plot.terminate_episode()


class HarvestAgent(Agent):
    """
    Agent Representation for RLLib
    """
    def __init__(self, name, index, char):
        super(HarvestAgent, self).__init__(name, index, char)

    @property
    def action_space(self):
        return Discrete(len(Actions))

    @property
    def observation_space(self):
        # TODO: Verify accuracy
        #       might be 22, 2
        import pdb; pdb.set_trace()
        return Box(
                low=0.0,
                high=0.0,
                shape=(len(GAME_ART), len(GAME_ART[0]), 1),
                dtype=np.float32)

