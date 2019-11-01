from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
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

parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument("--simple", action="store_true")


class CustomModel1(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Example of (optional) weight sharing between two different policies.
        # Here, we share the variables defined in the 'shared' variable scope
        # by entering it explicitly with tf.AUTO_REUSE. This creates the
        # variables for the 'fc1' layer in a global scope called 'shared'
        # outside of the policy's normal variable scope.
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


class CustomModel2(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Weights shared with CustomModel1
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env("test_discrete", lambda _: ArenaRllibEnv())
    single_env = ArenaRllibEnv()
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    single_env.close()

    # Each policy can have a different configuration (including custom model)
    def gen_policy(i):
        config = {
            "gamma": random.choice([0.95, 0.99]),
        }
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        "P{}".format(i): gen_policy(i)
        for i in range(2)
    }

    tune.run(
        "PPO",
        stop={"training_iteration": args.num_iters},
        config={
            "env": "test_discrete",
            "log_level": "DEBUG",
            "simple_optimizer": args.simple,
            "num_sgd_iter": 10,
            "multiagent": {
                "policies": policies,
            },
        },
    )
