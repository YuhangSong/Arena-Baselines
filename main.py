from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Simple example of using Multi-Agent and Hierarchical
(https://ray.readthedocs.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
from rllib to train an arena environment in ArenaRllibEnv.
"""

import argparse
import gym
import random

import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf

from envs_layer import ArenaRllibEnv

tf = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument("--env-id", type=str, default="Test-Discrete")


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env("arena_env",
                 lambda env_config: ArenaRllibEnv(env_config))

    single_env = ArenaRllibEnv({
        "env_id": args.env_id,
    })
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    number_agents = single_env.number_agents

    policy_id_prefix = 'policy'

    policies = {}
    for agent_i in range(number_agents):
        policy_id = "{}_{}".format(policy_id_prefix, agent_i)
        policies[policy_id] = (None, obs_space, act_space, {})

    agent_id_to_policy_id = {}
    for agent_i in range(number_agents):
        agent_id = "{}_{}".format(single_env.agent_id_prefix, agent_i)
        policy_id = "{}_{}".format(policy_id_prefix, agent_i)
        agent_id_to_policy_id[agent_id] = policy_id

    tune.run(
        "PPO",
        stop={"training_iteration": args.num_iters},
        config={
            "env": "arena_env",
            "env_config": {
                "env_id": args.env_id,
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: agent_id_to_policy_id[agent_id]
                ),
            },
        },
    )
