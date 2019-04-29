"""
Harvest.py
Jared Weinstein

INCOMPLETE implementation of a shared resource dilemma.
Agents must sustainably harvest apples that respawn in proportion
to how many exist
"""

import enum
import numpy as np
from pycolab import ascii_art

from gym.spaces import Box, Discrete
from .util.base_class import RewardSprite, Agent
from .util.social_dilemma import Dilemma

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
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PUNISH = 5

class HarvestEnvironment(Dilemma):

    def __init__(self):
        print("Harvest Environment is incomplete")
        raise NotImplementedError

        # create agents
        agents = []
        for i in range(2):
            name = 'agent-' + str(i)
            agent = HarvestAgent(name, i, str(i))
            agents.append(agent)

        # create pycolab sprite for each agent
        sprites = {}
        for agent in agents:
            sprites[agent.char] = ascii_art.Partial(self.HarvestSprite, agent.index, len(agents))

        super(HarvestEnvironment, self).__init__(GAME_ART, agents, sprites)


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


            # finished = True
            # for agent in things.values():
            #     if not agent.position[1] in [1, self.corner[1] - 2]:
            #         finished = False

            if self.index == 1:
                if self.position[1] == 1:
                    self.reward(the_plot, 1)
                    the_plot.terminate_episode()
                elif self.position[1] == (self.corner[1] - 2):
                    self.reward(the_plot, 100)
                    the_plot.terminate_episode()


            # # distribute reward
            # if finished == True:
            #     for agent in things.values():
            #         if agent.position[1] == 1:
            #             self.reward(the_plot, 1, agent.index)
            #         elif agent.position[1] == (self.corner[1] - 2):
            #             self.reward(the_plot, 100, agent.index)
            #
            #     the_plot.terminate_episode()


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
        nrow = len(GAME_ART)
        ncol = len(GAME_ART[0])
        high = len(set(''.join(GAME_ART))) - 1
        return Box(
                low=0,
                high=high,
                shape=(nrow, ncol, 1),
                dtype=np.float32)

