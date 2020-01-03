#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import arena
import argparse
import yaml
import copy
import os
import pickle
import logging
import utils

import numpy as np

import ray
from ray.tests.cluster_utils import Cluster
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments

# logger = logging.getLogger(__name__)


POLICY_ID_PREFIX = "policy"


def create_parser():
    """Returns parser with additional arena configs.
    """

    from ray.rllib.train import create_parser as create_parser
    parser = create_parser()

    parser.add_argument(
        "--is-shuffle-agents",
        action="store_true",
        help=(
            "Whether shuffle agents every episode. "
            "This helps the trained policies to have better generalization ability."
        ))
    parser.add_argument(
        "--train-mode",
        action="store_false",
        help="Whether run in train mode, with faster and smaller resulotion.")
    parser.add_argument(
        "--obs-type",
        default="visual_FP",
        type=str,
        help=(
            "Type of the observation; options: "
            "vector (low-dimensional vector observation); "
            "visual_FP (first-person visual observation); "
            "visual_TP (third-person visual observation); "
            "obs1-obs2-... (combine multiple types of observations); "
        ))
    parser.add_argument(
        "--iterations-per-reload",
        default=1,
        type=int,
        help=(
            "Number of iterations between each reload. "
            "In each reload, if num_populaton>1, one of the learning agents will be picked up randomly to face each other. "
            "In each reload, if policy_assignment==selfplay, the only one learning policy will be saved and all playing policies will be reloaded. "
        ))
    parser.add_argument(
        "--num-learning-policies",
        default="independent",
        type=str,
        help=(
            "How to assign policies to agents. Options are as follows: "
            "all (all agents are bound to learning policies, one for each. This is also known as independent learner. ); "
            "x (there are x agents bound to x learning policies, one for each; the other (num_agents-x) agents are bound to playing policies, one for each. Setting x=1 is known as selfplay. ); "
            "Playing policies donot explore or update, but they keep reloading weights from the current and previous learning policy at each reload. "
        ))
    parser.add_argument(
        "--selfplay-recent-prob",
        default=0.8,
        type=float,
        help=(
            "In selfplay, the probability of chosing recent model. "
            "In other cases, it will choose uniformly among historical models. "
        ))
    parser.add_argument(
        "--num-population",
        default=1,
        type=int,
        help=(
            "Number of learning agents in population-based training. "
            "In each reload, one of the learning agents will be picked up randomly to face each other. "
        ))

    return parser


def run(args, parser):

    # get config as experiments
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)

    else:
        input("# WARNING: it is recommended to use -f CONFIG.yaml, instead of passing args. Press enter to continue.")
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)
                ),
                "stop": args.stop,
                "config": dict(
                    args.config,
                    env=args.env,
                    env_config=dict(
                        is_shuffle_agents=args.is_shuffle_agents,
                        train_mode=args.train_mode,
                        obs_type=args.obs_type,
                    ),
                    iterations_per_reload=args.iterations_per_reload,
                    num_learning_policies=args.num_learning_policies,
                    selfplay_recent_prob=args.selfplay_recent_prob,
                    num_populaton=args.num_populaton,
                ),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    # expand experiments with grid_search, this is implemented to override
    # the default support of grid_search
    grid_experiments = {}
    for experiment_key in experiments.keys():
        for iterations_per_reload_item in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["iterations_per_reload"]):
            for num_learning_policies_item in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["num_learning_policies"]):
                for selfplay_recent_prob_item in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["selfplay_recent_prob"], experiments[experiment_key]["config"]["num_learning_policies"] == "all"):
                    for num_populaton_item in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["num_populaton"]):
                        grid_experiment_key = "{}_ipr={}_pa={}_srp={}_np={}".format(
                            experiment_key,
                            iterations_per_reload_item,
                            num_learning_policies_item,
                            selfplay_recent_prob_item,
                            num_populaton_item,
                        )
                        grid_experiments[grid_experiment_key] = copy.deepcopy(
                            experiments[experiment_key]
                        )
                        grid_experiments[grid_experiment_key]["config"]["iterations_per_reload"] = iterations_per_reload_item
                        grid_experiments[grid_experiment_key]["config"]["num_learning_policies"] = num_learning_policies_item
                        grid_experiments[grid_experiment_key]["config"]["selfplay_recent_prob"] = selfplay_recent_prob_item
                        grid_experiments[grid_experiment_key]["config"]["num_populaton"] = num_populaton_item

    experiments = grid_experiments

    for exp in experiments.values():

        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")
        if args.eager:
            exp["config"]["eager"] = True

        # generate config for arena
        if arena.is_all_arena_env(exp["env"]):

            # create dummy_env to get parameters/setting of env
            dummy_env = arena.ArenaRllibEnv(
                env=arena.get_one_from_grid_search(
                    arena.remove_arena_env_prefix(
                        exp["env"]
                    )
                ),
                env_config=exp["config"]["env_config"],
            )
            num_agents = dummy_env.number_agents

            agent_id_prefix = dummy_env.get_agent_id_prefix()

            def get_agent_i(agent_id):
                return int(agent_id.split(agent_id_prefix + "_")[1])

            # For now, we do not support using different spaces across agents
            # (i.e., all agents have to share the same brain in Arena-BuildingToolkit)
            # This is because we want to consider the transfer/sharing weight between agents.
            # If you do have completely different agents in game, one harmless work around is
            # to use the same brain, but define different meaning of the action in Arena-BuildingToolkit
            obs_space = dummy_env.observation_space
            act_space = dummy_env.action_space

            def get_policy_id(policy_i):
                return "{}_{}".format(POLICY_ID_PREFIX, policy_i)

            # create
            # policies: config of policies
            policies = {}
            # learning_policy_ids: a list of policy ids of which the policy is trained
            learning_policy_ids = []
            # playing_policy_ids: a list of policy ids of which the policy is not trained
            playing_policy_ids = []

            if exp["config"]["num_learning_policies"] == "all":
                num_learning_policies = num_agents
            else:
                num_learning_policies = int(
                    exp["config"]["num_learning_policies"])

            # build learning policies
            for agent_i in range(num_learning_policies):
                key_ = get_policy_id(agent_i)
                learning_policy_ids += [key_]
                policies[key_] = (
                    None, obs_space, act_space, {}
                )

            # build playing policy

            # build custom_action_dist to be playing mode dist (no exploration)
            # TODO: support pytorch policy and other algorithms, currently only add support for tf_action_dist on PPO
            # see this issue for a fix: https://github.com/ray-project/ray/issues/5729

            if exp["run"] not in ["PPO"]:
                raise NotImplementedError

            if act_space.__class__.__name__ == "Discrete":

                from ray.rllib.models.tf.tf_action_dist import Categorical
                from ray.rllib.utils.annotations import override

                class DeterministicCategorical(Categorical):
                    """Deterministic version of categorical distribution for discrete action spaces."""

                    @override(Categorical)
                    def _build_sample_op(self):
                        return tf.squeeze(tf.argmax(self.inputs, 1), axis=1)

                custom_action_dist = DeterministicCategorical

            elif act_space.__class__.__name__ == "Box":

                from ray.rllib.models.tf.tf_action_dist import Deterministic
                custom_action_dist = Deterministic

            else:

                raise NotImplementedError

            # build all other policies as playing policy
            for agent_i in range(num_learning_policies, num_agents):
                key_ = get_policy_id(agent_i)
                playing_policy_ids += [key_]
                policies[key_] = (
                    None, obs_space, act_space, {
                        "custom_action_dist": custom_action_dist
                    }
                )

            # create
            # policy_mapping_fn: a map from agent_id to policy_id

            # create policy_mapping_fn that maps agent i to policy i, so called policy_mapping_fn_i2i
            def policy_mapping_fn_i2i(agent_id):
                return get_policy_id(get_agent_i(agent_id))

            # use policy_mapping_fn_i2i as policy_mapping_fn
            policy_mapping_fn = policy_mapping_fn_i2i

            # create
            # on_train_result: a function called after each trained iteration

            input(exp["config"]["iterations_per_reload"])

            def on_train_result(info):
                if info["result"]["training_iteration"] % exp["config"]["iterations_per_reload"] == 0:
                    print(
                        "Reload playing policies and save learning policy.")
                    print(info["trainer"].logdir)

                    def get_checkpoint_path(population_i=None):

                        if population_i is None:
                            population_i = np.random.randint(
                                exp["config"]["num_populaton"]
                            )

                        checkpoint_path = os.path.join(
                            info["trainer"].logdir,
                            "learning_agent-p_{}-i_{}".format(
                                population_i,
                                info["trainer"].iteration,
                            )
                        )

                        return checkpoint_path

                    for policy_id in learning_policy_ids:
                        checkpoint_path = get_checkpoint_path()
                        try:
                            pickle.dump(
                                info["trainer"].get_policy(
                                    policy_id
                                ).get_weights(),
                                open(checkpoint_path, "wb")
                            )
                        except Exception as e:
                            print("Save learning agent p_{} i_{} failed: {}".format(
                                population_i,
                                info["trainer"].iteration,
                                e,
                            ))

                    for policy_id in (learning_policy_ids + playing_policy_ids):
                        checkpoint_path = get_checkpoint_path()
                        try:
                            info["trainer"].get_policy(
                                policy_id
                            ).set_weights(
                                pickle.load(open(checkpoint_path, "rb"))
                            )
                        except Exception as e:
                            print("Load learning agent p_{} i_{} failed: {}".format(
                                population_i,
                                info["trainer"].iteration,
                                e,
                            ))

            # generate multiagent part of the config
            if "multiagent" in exp["config"].keys():
                input("# WARNING: Override")

            exp["config"]["multiagent"] = {
                "policies": policies,
                "policy_mapping_fn": ray.tune.function(
                    policy_mapping_fn
                ),
                "policies_to_train": learning_policy_ids,
            }

            # generate callbacks part of the config
            if "callbacks" in exp["config"].keys():
                input("# WARNING: Override")

            # exp["config"]["callbacks"] = {
            #     "on_train_result": ray.tune.function(
            #         on_train_result
            #     ),
            # }

        # del customized configs, as these configs have been reflected on other configs
        del exp["config"]["iterations_per_reload"]
        del exp["config"]["num_learning_policies"]
        del exp["config"]["selfplay_recent_prob"]
        del exp["config"]["num_populaton"]

    # config ray cluster
    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory,
            )
        ray.init(
            address=cluster.redis_address,
        )
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus,
        )

    # run
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
