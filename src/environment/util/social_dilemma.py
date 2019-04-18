from ray.rllib.env import MultiAgentEnv
from pycolab import ascii_art

import numpy as np

class Dilemma(MultiAgentEnv):
    def __init__(self, art, agents, sprites):
        self.agents = agents
        self.game_art = art
        self.sprites = sprites
        self.game = self._make_game()

    def _make_game(self):
        """Builds and returns a sample environment"""

        # create mapping
        char_map = {}
        chars = sorted(set(''.join(self.game_art)))
        for i in range(len(chars)):
            char_map[ord(chars[i])] = i
        self.mapping = char_map


        return ascii_art.ascii_art_to_game(
                self.game_art,
                what_lies_beneath=' ',
                sprites = self.sprites
                )

    def _map_lower(self, observation):
        def f(x):
            return self.mapping.get(x)
        lower = [list(map(f, line)) for line in observation]
        normed = np.array(lower)
        new_shape = normed.shape + (1, )
        normed = normed.reshape(new_shape)
        return normed

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

        # print('stepping')

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

            obs = self._map_lower(observation)

            agent = self.agents[i]
            observations[agent.name] = obs
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
            observations[agent.name] = self._map_lower(observation.board.tolist())
        return observations
