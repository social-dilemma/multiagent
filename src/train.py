import argparse

import tensorflow as tf
import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env

from environment.example import ExampleEnvironment
from environment.prison import PrisonEnvironment
from models.conv_to_fc_net import ConvToFCNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'exp_name', None,
    'Name of the ray_results experiment directory where results are stored.')
tf.app.flags.DEFINE_string(
    'env', 'prison',
    'Name of the environment to rollout')
tf.app.flags.DEFINE_string(
    'algorithm', 'A3C',
    'Name of the rllib algorithm to use.')
tf.app.flags.DEFINE_integer(
    'num_agents', 2,
    'Number of agent policies')
tf.app.flags.DEFINE_integer(
    'train_batch_size', 30000,
    'Size of the total dataset over which one epoch is computed.')
tf.app.flags.DEFINE_integer(
    'checkpoint_frequency', 30,
    'Number of steps before a checkpoint is saved.')
tf.app.flags.DEFINE_integer(
    'training_iterations', 200,
    'Total number of steps to train for')
tf.app.flags.DEFINE_integer(
    'num_cpus', 4,
    'Number of available CPUs')
tf.app.flags.DEFINE_integer(
    'num_gpus', 0,
    'Number of available GPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpus_for_workers', False,
    'Set to true to run workers on GPUs rather than CPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpu_for_driver', False,
    'Set to true to run driver on GPU rather than CPU.')
tf.app.flags.DEFINE_float(
    'num_workers_per_device', 1,
    'Number of workers to place on a single device (CPU or GPU)')

hyperparameters = {
    'example': {
        'lr_init': 0.00136,
        'lr_final': 0.000028,
        'entropy_coeff': -.000687},
    'harvest': {
        'lr_init': 0.00136,
        'lr_final': 0.000028,
        'entropy_coeff': -.000687},
    'prison': {
        'lr_init': 0.00136,
        'lr_final': 0.000028,
        'entropy_coeff': -.000687}}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    if env == 'example':
        agents = ExampleEnvironment().agents
        def env_creator(_):
            return ExampleEnvironment()
    elif env == 'prison':
        can_punish = False
        manual_action = None
        agents = PrisonEnvironment(can_punish, manual_action).agents
        def env_creator(_):
            return PrisonEnvironment(can_punish, manual_action)
    else:
        print('unknown environment')
        raise NotImplementedError

    env_name = env + "_env"
    register_env(env_name, env_creator)

    # Each agent can have a different action and observation
    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for agent in agents:
        policy_graphs[agent.name] = (PPOPolicyGraph, agent.observation_space, agent.action_space, {})

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_to_fc_net"
    ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
                "train_batch_size": train_batch_size,
                "horizon": 1000,
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "gamma": 0.99, # Discount factor
                "entropy_coeff": hparams['entropy_coeff'],
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                "model": {"custom_model": "conv_to_fc_net", "use_lstm": True,
                          "lstm_cell_size": 128}

    })
    return algorithm, env_name, config

def main(unused_argv):
    del unused_argv

    ray.init(num_cpus=FLAGS.num_cpus)
    hparams = hyperparameters[FLAGS.env]

    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.algorithm,
                                      FLAGS.train_batch_size,
                                      FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device)

    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": FLAGS.training_iterations
            },
            'checkpoint_freq': FLAGS.checkpoint_frequency,
            "config": config,
        }
    })

if __name__ == "__main__":

    """ load arguments """
    parser = argparse.ArgumentParser(description='Train and play multiagent sequential dilemma')
    args = parser.parse_args()

    tf.app.run(main)

