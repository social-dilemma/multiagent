# Multi-Agent Cooperation in Sequential Social Dilemmas

This work is a implementation and exploration of current work in Multiagent
Reinforcement Learning (MARL). It is highly recommended that you read the following
two papers before diving in.
1. [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037)
2. [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647)

## Environments

Pycolab provides the abstraction for creating environments. This repository includes two functional
environments. ExampleEnvironment is a simple two player interaction. PrisonEnvironment is
a continuous two player prisoners dilemma

The play script allows you to quickly test an environment manually: an
extremely useful tool when debugging.

## Learning

Reinforcement Learning is handled by RLLib. Currently all training is done using
the A3C algorithm.

## Related Works

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

2.  Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzman, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K., Koster, R., Tina Zhu, Roff, H., Graepel, T. (2018). [Inequity aversion improves cooperation in intertemporal social dilemmas](https://arxiv.org/abs/1803.08884). In Advances in Neural Information Processing Systems (pp. 3330-3340).

3. Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z. & de Freitas, N. (2018). [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647). arXiv preprint arXiv:1810.08647.

4. Credit to [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) for providing a useful example of RLLib.
