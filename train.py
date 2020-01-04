#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import copy
import os
import pickle
import logging
import utils
import glob

import numpy as np

import ray
from ray.tests.cluster_utils import Cluster
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments
from ray.rllib.models.tf.tf_action_dist import Deterministic as DeterministiContinuous

import arena
from arena import DeterministicCategorical


class Trainer(ray.rllib.agents.trainer.Trainer):
    """Override Trainer so that it allow new configs."""
    _allow_unknown_configs = True


ray.rllib.agents.trainer.Trainer = Trainer

# logger = logging.getLogger(__name__)


def create_parser():
    """Returns parser with additional arena configs.
    """

    # import parser from rllib.train
    from ray.rllib.train import create_parser as create_parser
    parser = create_parser()

    parser.add_argument(
        "--is-shuffle-agents",
        action="store_true",
        help=(
            "Whether shuffle agents every episode. "
            "This helps the trained policies to have better generalization ability. "
        ))

    parser.add_argument(
        "--train-mode",
        action="store_false",
        default=True,
        help=(
            "Whether run Arena environments in train mode. "
            "In train mode, the Arena environments run in a faster clock and in smaller resulotion. "
        ))
    parser.add_argument(
        "--obs-type",
        default="visual_FP",
        type=str,
        help=(
            "Type of the observation. Options are as follows: "
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
            "In each reload, learning policies are saved and all policies are reloaded. "
        ))

    parser.add_argument(
        "--num-learning-policies",
        default="independent",
        type=str,
        help=(
            "How many agents in the game are bound to learning policies (one to each). Options are as follows: "
            "all (all agents are bound to learning policies, one for each. This is also known as independent learner.); "
            "x (there are x agents bound to x learning policies, one for each; the other (num_agents-x) agents are bound to playing policies, one for each.); "
            "Setting x=1 is known as selfplay. "
            "Playing policies donot explore or update, but they keep reloading weights from the current and previous learning policy at each reload. "
        ))

    parser.add_argument(
        "--selfplay-recent-prob",
        default=0.8,
        type=float,
        help=(
            "When reload, for playing policies only, the probability of chosing recent learning policy, against chosing uniformly among historical ones. "
        ))

    parser.add_argument(
        "--size-population",
        default=1,
        type=int,
        help=(
            "Number of policies to be trained in population-based training. "
            "In each reload, each one of all learning/player policies will be reloaded with one of the size_population policies randomly. "
        ))

    return parser


def run(args, parser):

    # get config as experiments
    if args.config_file:

        # load configs from yaml
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)

    else:

        # load configs from args
        input("# WARNING: it is recommended to use -f CONFIG.yaml, instead of passing args. Press hit enter to continue. ")

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
                    size_population=args.size_population,
                ),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    # expand experiments with grid_search, this is implemented to override
    # the default support of grid_search for customized configs
    grid_experiments = {}
    for experiment_key in experiments.keys():

        for env in arena.get_list_from_gridsearch(experiments[experiment_key]["env"]):

            experiments[experiment_key]["env"] = str(
                env
            )

            if arena.is_arena_env(env):
                # create dummy_env to get parameters/setting of env
                dummy_env = arena.ArenaRllibEnv(
                    env=arena.remove_arena_env_prefix(
                        experiments[experiment_key]["env"]
                    ),
                    env_config=experiments[experiment_key]["config"]["env_config"],
                )

                experiments[experiment_key]["config"]["num_agents"] = int(
                    dummy_env.number_agents
                )

                # For now, we do not support using different spaces across agents
                # (i.e., all agents have to share the same brain in Arena-BuildingToolkit)
                # This is because we want to consider the transfer/sharing weight between agents.
                # If you do have completely different agents in game, one harmless work around is
                # to use the same brain, but define different meaning of the action in Arena-BuildingToolkit
                experiments[experiment_key]["config"]["obs_space"] = copy.deepcopy(
                    dummy_env.observation_space
                )
                experiments[experiment_key]["config"]["act_space"] = copy.deepcopy(
                    dummy_env.action_space
                )

            for iterations_per_reload in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["iterations_per_reload"]):

                experiments[experiment_key]["config"]["iterations_per_reload"] = int(
                    iterations_per_reload
                )

                for num_learning_policies in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["num_learning_policies"]):

                    if num_learning_policies == "all":
                        experiments[experiment_key]["config"]["num_learning_policies"] = int(
                            experiments[experiment_key]["config"]["num_agents"]
                        )
                    else:
                        experiments[experiment_key]["config"]["num_learning_policies"] = int(
                            num_learning_policies
                        )

                    if arena.is_arena_env(env):

                        if experiments[experiment_key]["config"]["num_learning_policies"] < experiments[experiment_key]["config"]["num_agents"]:
                            is_selfplay = True
                        elif experiments[experiment_key]["config"]["num_learning_policies"] == experiments[experiment_key]["config"]["num_agents"]:
                            is_selfplay = False
                        else:
                            raise Exception(
                                "num_learning_policies can not be larger than num_agents."
                            )

                    else:

                        is_selfplay = False

                    experiments[experiment_key]["config"]["is_selfplay"] = bool(
                        is_selfplay
                    )

                    for selfplay_recent_prob in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["selfplay_recent_prob"], experiments[experiment_key]["config"]["is_selfplay"]):

                        if selfplay_recent_prob is None:
                            experiments[experiment_key]["config"]["selfplay_recent_prob"] = None
                        else:
                            experiments[experiment_key]["config"]["selfplay_recent_prob"] = float(
                                selfplay_recent_prob
                            )

                        for size_population in arena.get_list_from_gridsearch(experiments[experiment_key]["config"]["size_population"]):

                            experiments[experiment_key]["config"]["size_population"] = int(
                                size_population
                            )

                            grid_experiment_key = "{}_ipr={}_nlp={}_srp={}_sp={}".format(
                                experiment_key,
                                experiments[experiment_key]["config"]["iterations_per_reload"],
                                experiments[experiment_key]["config"]["num_learning_policies"],
                                experiments[experiment_key]["config"]["selfplay_recent_prob"],
                                experiments[experiment_key]["config"]["size_population"],
                            )

                            grid_experiments[grid_experiment_key] = copy.deepcopy(
                                experiments[experiment_key]
                            )

                            if not grid_experiments[grid_experiment_key].get("run"):
                                parser.error(
                                    "The following arguments are required: --run"
                                )
                            if not grid_experiments[grid_experiment_key].get("env") and not grid_experiments[grid_experiment_key].get("config", {}).get("env"):
                                parser.error(
                                    "The following arguments are required: --env"
                                )
                            if args.eager:
                                grid_experiments[grid_experiment_key]["config"]["eager"] = True

                            # generate config for arena
                            if arena.is_arena_env(grid_experiments[grid_experiment_key]["env"]):

                                # policies: config of policies
                                grid_experiments[grid_experiment_key]["config"]["multiagent"] = {
                                }
                                grid_experiments[grid_experiment_key]["config"]["multiagent"]["policies"] = {
                                }
                                # learning_policy_ids: a list of policy ids of which the policy is trained
                                grid_experiments[grid_experiment_key]["config"]["learning_policy_ids"] = [
                                ]
                                # playing_policy_ids: a list of policy ids of which the policy is not trained
                                grid_experiments[grid_experiment_key]["config"]["playing_policy_ids"] = [
                                ]

                                # create configs of learning policies
                                for agent_i in range(grid_experiments[grid_experiment_key]["config"]["num_learning_policies"]):
                                    key_ = arena.get_policy_id(agent_i)
                                    grid_experiments[grid_experiment_key]["config"]["learning_policy_ids"] += [
                                        key_
                                    ]
                                    grid_experiments[grid_experiment_key]["config"]["multiagent"]["policies"][key_] = (
                                        None,
                                        grid_experiments[grid_experiment_key]["config"]["obs_space"],
                                        grid_experiments[grid_experiment_key]["config"]["act_space"],
                                        {}
                                    )

                                del agent_i

                                if grid_experiments[grid_experiment_key]["run"] not in ["PPO"]:
                                    # build custom_action_dist to be playing mode dist (no exploration)
                                    # TODO: support pytorch policy and other algorithms, currently only add support for tf_action_dist on PPO
                                    # see this issue for a fix: https://github.com/ray-project/ray/issues/5729
                                    raise NotImplementedError

                                # create configs of playing policies
                                for agent_i in range(grid_experiments[grid_experiment_key]["config"]["num_learning_policies"], experiments[experiment_key]["config"]["num_agents"]):
                                    key_ = arena.get_policy_id(agent_i)
                                    grid_experiments[grid_experiment_key]["config"]["playing_policy_ids"] += [
                                        key_]
                                    grid_experiments[grid_experiment_key]["config"]["multiagent"]["policies"][key_] = (
                                        None,
                                        grid_experiments[grid_experiment_key]["config"]["obs_space"],
                                        grid_experiments[grid_experiment_key]["config"]["act_space"],
                                        {
                                            "custom_action_dist": {
                                                "Discrete": DeterministicCategorical,
                                                "Box": DeterministiContinuous
                                            }[
                                                grid_experiments[grid_experiment_key]["config"]["act_space"].__class__.__name__
                                            ]
                                        }
                                    )

                                # policy_mapping_fn: a map from agent_id to policy_id
                                # use policy_mapping_fn_i2i as policy_mapping_fn
                                grid_experiments[grid_experiment_key]["config"]["multiagent"]["policy_mapping_fn"] = ray.tune.function(
                                    arena.policy_mapping_fn_i2i
                                )

                                # on_train_result: a function called after each trained iteration
                                def on_train_result(info):

                                    if info["result"]["training_iteration"] % info["trainer"].config["iterations_per_reload"] == 0:

                                        print(
                                            "Save learning policy and reload all policies."
                                        )

                                        def get_checkpoint_path(population_i=None, iteration_i=None):
                                            """Get checkpoint_path from population_i and iteration_i.
                                            """

                                            # if population_i is None, generate one
                                            if population_i is None:
                                                population_i = np.random.randint(
                                                    info["trainer"].config["size_population"]
                                                )

                                            # if iteration_i is None, use info["trainer"].iteration, i.e., the latest iteration
                                            if iteration_i is None:
                                                iteration_i = info["trainer"].iteration

                                            checkpoint_path = os.path.join(
                                                info["trainer"].logdir,
                                                "learning_agent/p_{}-i_{}".format(
                                                    population_i,
                                                    iteration_i,
                                                )
                                            )

                                            return checkpoint_path, population_i, iteration_i

                                        # save learning policies
                                        for policy_id in info["trainer"].config["learning_policy_ids"]:

                                            policy_i = info["trainer"].get_policy(
                                                policy_id
                                            )

                                            # check if policy_i has population_i assigned
                                            if hasattr(policy_i, "population_i"):
                                                # if so, get it
                                                population_i = policy_i.population_i
                                            else:
                                                # if not, set population_i to None
                                                # it will later be assigned
                                                population_i = None

                                            # get checkpoint_path
                                            checkpoint_path, population_i, iteration_i = get_checkpoint_path(
                                                population_i=population_i
                                            )

                                            # check checkpoint_path exists, if not, create one
                                            if not os.path.exists(os.path.dirname(checkpoint_path)):
                                                try:
                                                    os.makedirs(
                                                        os.path.dirname(
                                                            checkpoint_path)
                                                    )
                                                except OSError as exc:
                                                    # Guard against race condition
                                                    if exc.errno != errno.EEXIST:
                                                        raise

                                            # save to checkpoint_path
                                            try:
                                                pickle.dump(
                                                    info["trainer"].get_policy(
                                                        policy_id
                                                    ).get_weights(),
                                                    open(checkpoint_path, "wb")
                                                )
                                                print("Save learning policy {} in population {} at iteration {} succeed".format(
                                                    policy_id,
                                                    population_i,
                                                    iteration_i,
                                                ))
                                            except Exception as e:
                                                print("Save learning policy {} in population {} at iteration {} failed: {}".format(
                                                    policy_id,
                                                    population_i,
                                                    iteration_i,
                                                    e,
                                                ))

                                        del policy_id

                                        def remove_iteration_i_in_checkpoint_path(checkpoint_path):
                                            return checkpoint_path.split("-i_")[0] + "-i_"

                                        # reload all policies
                                        for policy_id in (info["trainer"].config["learning_policy_ids"] + info["trainer"].config["playing_policy_ids"]):

                                            policy_i = info["trainer"].get_policy(
                                                policy_id
                                            )

                                            # get checkpoint_path
                                            checkpoint_path, population_i, iteration_i = get_checkpoint_path()

                                            checkpoint_path_without_iteration_i = remove_iteration_i_in_checkpoint_path(
                                                checkpoint_path
                                            )

                                            # get possible_iterations
                                            possible_iterations = []
                                            for file in glob.glob(checkpoint_path_without_iteration_i + "*"):
                                                possible_iterations += [
                                                    int(
                                                        file.split(
                                                            checkpoint_path_without_iteration_i
                                                        )[1]
                                                    )
                                                ]

                                            del file

                                            possible_iterations = np.asarray(
                                                possible_iterations
                                            )
                                            possible_iterations.sort()

                                            if policy_id in info["trainer"].config["learning_policy_ids"]:
                                                # for learning policy, it only reload to the recent checkpoint
                                                load_recent_prob = 1.0
                                            elif policy_id in info["trainer"].config["playing_policy_ids"]:
                                                # for playing policy, it reload according to selfplay_recent_prob
                                                load_recent_prob = float(
                                                    info["trainer"].config["selfplay_recent_prob"]
                                                )
                                            else:
                                                raise NotImplementedError

                                            principle = str(
                                                np.random.choice(
                                                    ["recent", "uniform"],
                                                    replace=False,
                                                    p=[
                                                        load_recent_prob,
                                                        1.0 - load_recent_prob
                                                    ]
                                                )
                                            )

                                            if principle in ["recent"]:
                                                iteration_i = possible_iterations[-1]
                                            elif principle in ["uniform"]:
                                                iteration_i = np.random.choice(
                                                    possible_iterations,
                                                    replace=False,
                                                )
                                            else:
                                                raise NotImplementedError

                                            # get checkpoint_path, population_i is re-generated, iteration_i is specified
                                            checkpoint_path, population_i, iteration_i = get_checkpoint_path(
                                                population_i=None,
                                                iteration_i=iteration_i,
                                            )

                                            try:
                                                policy_i.set_weights(
                                                    pickle.load(
                                                        open(
                                                            checkpoint_path, "rb")
                                                    )
                                                )
                                                policy_i.population_i = population_i
                                                print("Load {} policy {} in population {} at iteration {} succeed. A result of load_recent_prob={} with principle={}".format(
                                                    "learning" if policy_id in info["trainer"].config[
                                                        "learning_policy_ids"] else "playing",
                                                    policy_id,
                                                    population_i,
                                                    iteration_i,
                                                    load_recent_prob,
                                                    principle,
                                                ))
                                            except Exception as e:
                                                print("Load {} policy {} in population {} at iteration {} failed: {}".format(
                                                    "learning" if policy_id in info["trainer"].config[
                                                        "learning_policy_ids"] else "playing",
                                                    policy_id,
                                                    population_i,
                                                    iteration_i,
                                                ))

                                        del policy_id

                                grid_experiments[grid_experiment_key]["config"]["callbacks"] = {
                                }
                                grid_experiments[grid_experiment_key]["config"]["callbacks"]["on_train_result"] = ray.tune.function(
                                    on_train_result
                                )

                                grid_experiments[grid_experiment_key]["config"]["multiagent"]["policies_to_train"] = copy.deepcopy(
                                    grid_experiments[grid_experiment_key]["config"]["learning_policy_ids"]
                                )

                        del size_population

                    del selfplay_recent_prob

                del num_learning_policies

            del iterations_per_reload

        del env

    del experiment_key

    # config ray cluster
    if args.ray_num_nodes:
        cluster = Cluster()
        for ray_node in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory,
            )
        del ray_node
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

    experiments_to_run = copy.deepcopy(grid_experiments)

    # delete grid_experiments and experiments so that the functions passed to tune
    # via config cannot have access to them
    del grid_experiments
    del experiments

    # run experiments
    run_experiments(
        experiments_to_run,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
