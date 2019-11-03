from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Simple example of using Multi-Agent and Hierarchical
(https://ray.readthedocs.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
from rllib to train an arena environment in ArenaRllibEnv.
"""

import argparse
import random
import time

import numpy as np

import ray
from ray import tune
from ray.rllib.utils import try_import_tf

from envs_layer import ArenaRllibEnv

tf = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--env-id", type=str,
                    default="Tennis-Sparse-2T1P-Discrete")
parser.add_argument("--policy-assignment", type=str, default="independent",
                    help="independent (independent learners), self-play (one policy, only one agent is learning, the others donot explore)")
parser.add_argument("--num-iters", type=int, default=1000)
parser.add_argument("--num-cpus-total", type=int, default=12)
parser.add_argument("--num-gpus-total", type=int, default=2)

policy_id_prefix = "policy"

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    env_config = {
        "env_id": args.env_id,
    }

    dummy_env = ArenaRllibEnv(env_config)
    number_agents = dummy_env.number_agents

    # For now, we do not support using different spaces across agents
    # (i.e., all agents have to share the same brain in Arena-BuildingToolkit)
    # This is because we want to consider the transfer/sharing weight between agents.
    # If you do have completely different agents in game, one harmless work around is
    # to use the same brain, but define different meaning of the action in Arena-BuildingToolkit
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    def get_policy_id(policy_i):
        return "{}_{}".format(policy_id_prefix, policy_i)

    # create config of policies
    policies = {}

    if args.policy_assignment in ["independent"]:
        # build number_agents policies
        for agent_i in range(number_agents):
            policy_id = get_policy_id(agent_i)
            policies[policy_id] = (None, obs_space, act_space, {})
    elif args.policy_assignment in ["self-play"]:
        # build just one learning policy
        policies["policy_learning_0"] = (None, obs_space, act_space, {})
        # and all other policies are playing policy
        for agent_i in range(1, number_agents):
            policies["policy_playing_{}".format(agent_i)] = (
                None, obs_space, act_space, {"custom_action_dist": "xxx"})
    else:
        raise NotImplementedError

    # create a map from agent_id to policy_id
    agent_id_to_policy_id = {}

    for agent_i in range(number_agents):
        agent_id = dummy_env.get_agent_id(agent_i)
        if args.policy_assignment in ["independent"]:
            # each agent is assigned with a independent policy
            policy_id = get_policy_id(agent_i)
            agent_id_to_policy_id[agent_id] = policy_id
        elif args.policy_assignment in ["self-play"]:
            # all agents are assigned with the same policy
            policy_id = get_policy_id(0)
            agent_id_to_policy_id[agent_id] = policy_id
        else:
            raise NotImplementedError

    # check if all agent_id are covered in agent_id_to_policy_id
    for agent_id in dummy_env.get_agent_ids():
        if agent_id not in agent_id_to_policy_id.keys():
            raise Exception("All agent_id has to be mentioned in agent_id_to_policy_id.keys(). \
                agent_id of {} is not mentioned".format(agent_id))

    tune.run(
        "PPO",
        stop={"training_iteration": args.num_iters},
        config={
            "env": "arena_env",
            "env_config": env_config,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: agent_id_to_policy_id[agent_id]
                ),
            },

            # atari-ppo
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.1,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "sample_batch_size": 100,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "vf_share_layers": True,

            # === Resources ===
            # Number of GPUs to allocate to the trainer process. Note that not all
            # algorithms can take advantage of trainer GPUs. This can be fractional
            # (e.g., 0.3 GPUs).
            "num_gpus": args.num_gpus_total,
            # for arena_env scaling up with num_workers is tested to be better than
            # scaling up with num_envs_per_worker,
            # so set num_workers = args.num_cpus_total - 1 (1 for num_cpus_for_driver)
            "num_workers": args.num_cpus_total - 1,
        },
    )
