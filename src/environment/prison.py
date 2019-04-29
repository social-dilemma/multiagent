"""
Prison.py
Jared Weinstein

An extended implementation of the Prisoner's Dilemma
The right side of the board represents cooperative behavior
The left side of the board is selfish defective behavior
"""

import enum
import numpy as np
import collections

from pycolab import ascii_art
from ray.rllib.env import MultiAgentEnv

from gym.spaces import Box, Discrete
from .util.social_dilemma import Dilemma
from .util.base_class import RewardSprite, Agent

GAME_ART = ['#    0    #',
            '#    1    #']

class Actions(enum.IntEnum):
    """ Actions for the player """
    STAY = 0
    PUNISH = 1
    RIGHT = 2
    LEFT = 3

class Action_Mode(enum.IntEnum):
    COOPERATE = 0
    DEFECT = 1
    TRAINED = 2
    TIT_TAT = 3

class PrisonEnvironment(Dilemma):
    def __init__(self, can_punish=False, overide_action=None):
        # create agents
        agents = []
        for i in range(2):
            name = 'agent-' + str(i)
            agent = PrisonAgent(name, i, str(i))
            agents.append(agent)

        # create pycolab sprite for each agent
        sprites = {}
        for i in range(len(agents)):
            agent = agents[i]
            action_mode = Action_Mode.TRAINED

            # manually override agent-0 actions
            if i == 0 and overide_action is not None:
                action_mode = overide_action

            sprites[agent.char] = ascii_art.Partial(self.PrisonSprite,
                                                    agent.index,
                                                    len(agents),
                                                    can_punish,
                                                    action_mode)

        super(PrisonEnvironment, self).__init__(GAME_ART, agents, sprites)


    class PrisonSprite(RewardSprite):
        COOPERATION_MULTIPLE = .75
        SELFISH_MULTIPLE = 1
        REWARD_PERIOD = 10
        HISTORY_LEN = 3

        """ Sprite representing player for Pycolab """

        def __init__(self, corner, position, character, index, n_unique, can_punish=False, action_mode="trained"):
            self.step = 0
            self.can_punish = can_punish
            self.action_mode = action_mode

            if action_mode == Action_Mode.TIT_TAT:
                self.move_history = collections.deque(
                        self.HISTORY_LEN * [Actions.STAY], self.HISTORY_LEN)

            super(PrisonEnvironment.PrisonSprite, self).__init__(corner, position, character, index, n_unique)

        def update(self, actions, board, layers, backdrop, things, the_plot):
            action = self.action(actions, self.index)
            # del layers, backdrop, things, actions

            self.step += 1
            opponent = self.index ^ 1

            # override action if necessary
            if self.action_mode == Action_Mode.COOPERATE:
                action = Actions.RIGHT
            elif self.action_mode == Action_Mode.DEFECT:
                action = Actions.LEFT
            elif self.action_mode == Action_Mode.TIT_TAT:
                opponent_action = self.action(actions, opponent)
                self.move_history.append(opponent_action)
                action = self.move_history[0]

            # action update
            if action == Actions.LEFT:
                self._west(board, the_plot)
            elif action == Actions.RIGHT:
                self._east(board, the_plot)
            elif self.can_punish and action == Actions.PUNISH:
                self.reward(the_plot, -1, opponent)


            # calculate reward
            if self.step % self.REWARD_PERIOD == 0:
                my_pos = things[str(self.index)].position[1] - 1
                op_pos = things[str(opponent)].position[1] - 1
                max_pos = len(GAME_ART[0]) - 3

                my_cooperation = float(my_pos) / max_pos
                op_cooperation = float(op_pos) / max_pos

                reward = 0
                reward += (my_cooperation + op_cooperation) * self.COOPERATION_MULTIPLE
                reward += (1 - my_cooperation) * self.SELFISH_MULTIPLE
                self.reward(the_plot, reward)

class PrisonAgent(Agent):
    """
    Agent Representation for RLLib
    """
    def __init__(self, name, index, char):
        super(PrisonAgent, self).__init__(name, index, char)

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
