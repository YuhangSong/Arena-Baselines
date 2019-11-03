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
import copy

import numpy as np

import ray
from ray import tune
from ray.rllib.utils import try_import_tf

from envs_layer import ArenaRllibEnv

tf = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--env-id", type=str,
                    default="Tennis-Sparse-2T1P-Discrete")
parser.add_argument("--policy-assignment", type=str, default="self-play",
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

        # build number_agents independent learning policies
        for agent_i in range(number_agents):
            policies[get_policy_id(agent_i)] = (None, obs_space, act_space, {})

    elif args.policy_assignment in ["self-play"]:

        # build just one learning policy
        policies[get_policy_id(0)] = (None, obs_space, act_space, {})

        # build all other policies as playing policy

        # build custom_action_dist to be playing mode dist (no exploration)
        # TODO: support pytorch policy, currently only add support for tf_action_dist
        if act_space.__class__.__name__ == "Discrete":

            from agents_layer import DeterministicCategorical
            custom_action_dist = DeterministicCategorical

        elif act_space.__class__.__name__ == "Box":

            from ray.rllib.models.tf.tf_action_dist import Deterministic
            custom_action_dist = Deterministic

        else:

            raise NotImplementedError

        # build all other policies as playing policy
        for agent_i in range(1, number_agents):
            policies[get_policy_id(agent_i)] = (
                None, obs_space, act_space, {"custom_action_dist": custom_action_dist})

    else:
        raise NotImplementedError

    # create a map from agent_id to policy_id
    if args.policy_assignment in ["independent", "self-play"]:

        # create policy_mapping_fn that maps agent i to policy i, so called policy_mapping_fn_i2i
        agent_id_prefix = dummy_env.get_agent_id_prefix()

        def get_agent_i(agent_id):
            return int(agent_id.split(agent_id_prefix + "_")[1])

        def policy_mapping_fn_i2i(agent_id):
            return get_policy_id(get_agent_i(agent_id))

        # use policy_mapping_fn_i2i as policy_mapping_fn
        policy_mapping_fn = policy_mapping_fn_i2i

    else:
        raise NotImplementedError

    tune.run(
        "PPO",
        stop={"training_iteration": args.num_iters},
        config={
            "env": args.env_id if "NoFrameskip" in args.env_id else "arena_env",
            "env_config": env_config,
            "multiagent": {
                "policies": {} if "NoFrameskip" in args.env_id else policies,
                "policy_mapping_fn": None if "NoFrameskip" in args.env_id else policy_mapping_fn,
            },

            # use atari-ppo settings
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
