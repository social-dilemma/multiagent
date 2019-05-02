# Multi-Agent Cooperation in Sequential Social Dilemmas

(More Information)[http://zoo.cs.yale.edu/classes/cs490/18-19b/weinstein.jared.jtw37]

This work is an implementation and exploration of current work in Multiagent Reinforcement Learning (MARL). It is highly recommended that you read the following two papers before diving in.
1. [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037)
2. [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647)

## Quick Start

1. Switch to your virtual env
2. `pip install -r requirements.txt`
3. `python train.py`
4. `python test.py ~/ray_results/prison_A3C/[training_instance]/
   [checkpoint_num]`

Training results are usually saved in your ray_results directory located in the root directory


## Environments

Pycolab provides the abstraction for creating environments. Although this repository includes three environments, only the PrisonEnvironment has been fully developed and tested.

The PrisonEnvironment instantiates a gridworld variant of the classic Prisoner's
Dilemma. At each step of the game, both agents independently choose to move left,
move right, or stay still. The left side of the board represents full defection
and the right side of the board represents full cooperation. Intermediate
positions are a linear combination of the extremes. Rewards are distributed every
10 timesteps of the game. The figure below shows the corresponding rewards for
four primary states of the game.

![Alt text](resources/prisoners_gamestate.png?raw=true "4 primary game states
and corresponding rewards"){:height="50%" width="50%"}

`python play.py` allows you to quickly run a manual version of the game. The
script is extremely helpful when debugging the environment alone.


## Learning

Reinforcement Learning is handled by RLLib. Currently all training is done using
the A3C algorithm.

## Unresolved Issues

The test.py script has an issue where multiple asynchronous agents simultaneousy
interact with the environment. This messes up the visualization. One potential
solution is to wait until the interactions with the environment have finished
before starting the game. This only takes a few seconds. The better solution
would be to fix the issue and submit a PR!

## Related Works

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

2.  Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzman, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K., Koster, R., Tina Zhu, Roff, H., Graepel, T. (2018). [Inequity aversion improves cooperation in intertemporal social dilemmas](https://arxiv.org/abs/1803.08884). In Advances in Neural Information Processing Systems (pp. 3330-3340).

3. Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z. & de Freitas, N. (2018). [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647). arXiv preprint arXiv:1810.08647.

4. Credit to [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) for providing a useful example of RLLib.
