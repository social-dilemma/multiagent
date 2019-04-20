from pycolab.prefab_parts import sprites
import numpy as np

NO_ACTION = 0

class RewardSprite(sprites.MazeWalker):
    """ Sprite representing any player with unique actions and rewards """

    def __init__(self, corner, position, character, index, n_unique):
        self.index = index
        self.n_unique = n_unique

        super(RewardSprite, self).__init__(corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        raise NotImplementedError

    def reward(self, plot, value, index=None):
        if index == None: index = self.index
        reward = np.zeros(self.n_unique)
        reward[index] = value
        plot.add_reward(reward)

    def action(self, actions, index):
        if actions == None:
            action = NO_ACTION
        else:
            action = actions[index]
        return action

class Agent():
    """
    Representation of any artificial actor to be trained. (RLLib)
    """

    def __init__(self, name, index, char):
        """
        Input:
        -------------
        1. name (String)
           Unique identifier used by RLLib
        2. index (Int)
           Unique index used by pycolab
        3. char (Char)
           Character for the pycolab ascii bitmap

        """

        self.name = name
        self.index = index
        self.char = char

    @property
    def action_space(self):
        """
        Possible actions taken by the agent. This becomes the output
        of our Neural Net

        Returns:
        -------------
        actions (gym.spaces)
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Observation size of the agent. Input size of NN

        Returns:
        -------------
        obs_space (gym.spaces)
        """
        raise NotImplementedError



