import json
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pathlib import Path
import ray
import argparse
from ray.cloudpickle import cloudpickle
from ray.rllib.agents.registry import get_agent_class

from models.conv_to_fc_net import ConvToFCNet


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path / 'params.json' # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""

    pklfile = path / 'params.pkl'  # params.pkl is the config file
    with open(pklfile, 'rb') as file:
        pkldata = cloudpickle.load(file)
    return pkldata

def run(args):
    result_dir = Path(args.result_dir)
    config = get_rllib_config(result_dir)
    pkl = get_rllib_pkl(result_dir)

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policy_graphs', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        print('something went wrong with multiagent policy graph')
        import pdb; pdb.set_trace()

    # Create and register a gym+rllib env
    env_creator = pkl['env_config']['func_create']
    env_name = config['env_config']['env_name']
    register_env(env_name, env_creator.func)

    ModelCatalog.register_custom_model("conv_to_fc_net", ConvToFCNet)
    # Determine agent and checkpoint
    config_run = config['env_config']['run']
    agent_cls = get_agent_class(config_run)

    # Run on only one cpu for rendering purposes if possible; A3C requires two
    if config_run == 'A3C':
        config['num_workers'] = 1
    else:
        config['num_workers'] = 0

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir / ('checkpoint_' + args.checkpoint_num)
    checkpoint = checkpoint / ('checkpoint-' + args.checkpoint_num)
    print('Loading checkpoint', checkpoint)
    agent.restore(str(checkpoint))
    env = agent.local_evaluator.env

    policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
    mapping_cache = {}
    policy_map = agent.local_evaluator.policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    steps = 0
    while steps < (config['horizon'] or steps + 1):
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (config['horizon'] or steps + 1):
            action_dict = {}
            for agent_id in state.keys():
                a_state = state[agent_id]
                if a_state is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state_init, _ = agent.compute_action(
                            a_state,
                            state=state_init[policy_id],
                            policy_id=policy_id)
                        state_init[policy_id] = p_state_init
                    else:
                        a_action = agent.compute_action(
                            a_state, policy_id=policy_id)
                    action_dict[agent_id] = a_action

            import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Evaluates a reinforcement learning agent '
                'given a checkpoint.')

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')
    args = parser.parse_args()
    ray.init(num_cpus=1)
    run(args)
