import os
import time
import logging
import glob
import pickle
import gym
import ray

from copy import deepcopy as dcopy

import numpy as np

from .envs import *
from .models import *
from .utils import *
from .constants import *
from .arguments import *

logger = logging.getLogger(__name__)


def policy_mapping_fn_i2i(agent_id):
    """A policy_mapping_fn that maps agent i to policy i.
    """
    return policy_i2id(agent_id2i(agent_id))


def is_arena_env(each_env):
    """Check if a env (string) is an arena env.
    """
    return each_env[:len(ARENA_ENV_PREFIX)] == ARENA_ENV_PREFIX


def is_all_arena_env(env):
    """Check if all env in a grid_search env are arena env.
    If env is not a grid_search, return is_arena_env.
    """
    if is_grid_search(env):
        for each_env in env["grid_search"]:
            if not is_arena_env(each_env):
                return False
        return True
    else:
        return is_arena_env(env)


def is_any_arena_env(env):
    """Check if any env in a grid_search env is arena env.
    If env is not a grid_search, return is_arena_env.
    """
    if is_grid_search(env):
        for each_env in env["grid_search"]:
            if is_arena_env(each_env):
                return True
        return False
    else:
        return is_arena_env(env)


def get_checkpoint_path(logdir, population_i, iteration_i):
    """Get checkpoint_path from population_i and iteration_i.
    """

    checkpoint_path = os.path.join(
        logdir,
        "{}{}{}-{}{}".format(
            CHECKPOINT_PATH_PREFIX,
            CHECKPOINT_PATH_POPULATION_PREFIX,
            population_i,
            CHECKPOINT_PATH_ITERATION_PREFIX,
            iteration_i,
        )
    )

    return checkpoint_path


def checkpoints_2_checkpoint_paths(checkpoints):
    """Convert from checkpoints to checkpoint_paths.
    Example:
        Arguments:
            checkpoints: {
                policy_0:
                    logdir_0:
                        population_0:
                            [
                                iteration_0,
                                iteration_1,
                                ...,
                            ]
                        population_1:
                            ...,
                    logdir_1:
                        ...,
                policy_1:
                    ...,
            }
        Returns:
            {
                policy_0:
                    [
                        logdir_0/population_0/iteration_0,
                        logdir_0/population_0/iteration_1,
                        ...,
                        logdir_0/population_1/iteration_0,
                        logdir_0/population_1/iteration_1,
                        ...,
                        logdir_1/population_0/iteration_0,
                        logdir_1/population_0/iteration_1,
                        ...,
                        logdir_1/population_1/iteration_0,
                        logdir_1/population_1/iteration_1,
                        ...,
                    ],
                policy_1:
                    [
                        ...,
                    ],
            }
    """

    checkpoint_paths = {}
    for policy_id in checkpoints.keys():

        checkpoint_paths[policy_id] = []
        for logdir in checkpoints[policy_id]:

            for population_i in checkpoints[policy_id][logdir]:

                for iteration_i in checkpoints[policy_id][logdir][population_i]:

                    checkpoint_path = get_checkpoint_path(
                        logdir=logdir,
                        population_i=population_i,
                        iteration_i=iteration_i,
                    )
                    checkpoint_paths[policy_id] += [checkpoint_path]

    return checkpoint_paths


def get_possible_logdirs(base_logdir="~/ray_results/"):
    """Get possible logdirs on this machine.
    """

    # replace ~ in base_logdir with os.path.expanduser("~"),
    # so that grob can work properly
    base_logdir = base_logdir.replace(
        "~",
        os.path.expanduser("~"),
    )

    possible_logdirs = []
    for logdirs_level_0 in glob.glob("{}/Arena-Benchmark{}".format(base_logdir, "*")):
        for logdirs_level_1 in glob.glob("{}/{}".format(logdirs_level_0, "*")):
            if logdirs_level_1[-5:] != ".json":
                possible_logdirs += [logdirs_level_1]

    return possible_logdirs


def get_possible_populations(logdir):
    """Get possible populations on the disk, sorted in order
    """

    possible_populations = []
    checkpoint_path_search_prefix = os.path.join(
        logdir,
        "{}{}".format(
            CHECKPOINT_PATH_PREFIX,
            CHECKPOINT_PATH_POPULATION_PREFIX,
        )
    )
    for file in glob.glob(checkpoint_path_search_prefix + "*"):
        population_i = int(
            file.split(
                checkpoint_path_search_prefix
            )[1].split("-")[0]
        )
        if population_i not in possible_populations:
            possible_populations += [
                population_i
            ]
    possible_populations = np.asarray(
        possible_populations
    )
    possible_populations.sort()
    return list(possible_populations)


def get_possible_iterations(logdir, population_i):
    """Get possible iterations on the disk, sorted in order
    """

    possible_iterations = []
    checkpoint_path_search_prefix = os.path.join(
        logdir,
        "{}{}{}-{}".format(
            CHECKPOINT_PATH_PREFIX,
            CHECKPOINT_PATH_POPULATION_PREFIX,
            population_i,
            CHECKPOINT_PATH_ITERATION_PREFIX,
        )
    )
    for file in glob.glob(checkpoint_path_search_prefix + "*"):
        possible_iterations += [
            int(
                file.split(
                    checkpoint_path_search_prefix
                )[1].split("-")[0]
            )
        ]
    possible_iterations = np.asarray(
        possible_iterations
    )
    possible_iterations.sort()
    return list(possible_iterations)


def get_possible_iteration_indexes(logdir, population_i):
    possible_iterations = get_possible_iterations(logdir, population_i)
    possible_iteration_indexes = range(len(possible_iterations))
    return possible_iteration_indexes, possible_iterations


def on_train_result(info):
    """Function called after each trained iteration
    """

    if info["result"]["training_iteration"] % info["trainer"].config["iterations_per_reload"] == 0:

        logger.info(
            "Save learning policy and reload all policies."
        )

        # save learning policies
        for policy_id in info["trainer"].config["learning_policy_ids"]:

            policy = info["trainer"].get_policy(
                policy_id
            )

            # check if policy has population_i assigned
            if hasattr(policy, "population_i"):
                # if so, get it
                population_i = policy.population_i
            else:
                possible_populations = get_possible_populations(
                    logdir=info["trainer"].logdir,
                )
                # if not, generate one from those have not been used
                population_i = np.random.choice(
                    list_subtract(
                        range(
                            info["trainer"].config["size_population"]
                        ),
                        possible_populations
                    )
                )

            iteration_i = info["trainer"].iteration

            # get checkpoint_path by population_i and recent iteration
            checkpoint_path = get_checkpoint_path(
                logdir=info["trainer"].logdir,
                population_i=population_i,
                iteration_i=iteration_i,
            )

            prepare_path(checkpoint_path)

            save_message = "Save learning policy {} in population {} at iteration {}".format(
                policy_id,
                population_i,
                iteration_i,
            )

            # save to checkpoint_path
            try:
                pickle.dump(
                    info["trainer"].get_policy(
                        policy_id
                    ).get_weights(),
                    open(
                        checkpoint_path,
                        "wb"
                    )
                )
                logger.info("{} succeed".format(
                    save_message,
                ))
            except Exception as e:
                logger.warning("{} failed: {}.".format(
                    save_message,
                    e,
                ))

        possible_populations = get_possible_populations(
            logdir=info["trainer"].logdir,
        )

        # reload all policies
        for policy_id in (info["trainer"].config["learning_policy_ids"] + info["trainer"].config["playing_policy_ids"]):

            policy = info["trainer"].get_policy(
                policy_id
            )

            population_i = np.random.choice(
                range(info["trainer"].config["size_population"]),
            )

            load_message = "Load {} policy {} in population {}".format(
                {
                    True: "learning",
                    False: "playing",
                }[
                    policy_id in info["trainer"].config["learning_policy_ids"]
                ],
                policy_id,
                population_i,
            )

            if population_i in possible_populations:

                # get possible_iterations
                possible_iterations = get_possible_iterations(
                    logdir=info["trainer"].logdir,
                    population_i=population_i,
                )

                if policy_id in info["trainer"].config["learning_policy_ids"]:
                    # for learning policy, it only reload to the recent checkpoint
                    load_recent_prob = 1.0
                elif policy_id in info["trainer"].config["playing_policy_ids"]:
                    # for playing policy, it reload according to playing_policy_load_recent_prob
                    load_recent_prob = info["trainer"].config["playing_policy_load_recent_prob"]
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
                checkpoint_path = get_checkpoint_path(
                    logdir=info["trainer"].logdir,
                    population_i=population_i,
                    iteration_i=iteration_i,
                )

                load_message = "{} at iteration {}".format(
                    load_message,
                    iteration_i,
                )

                try:
                    policy.set_weights(
                        pickle.load(
                            open(
                                checkpoint_path,
                                "rb"
                            )
                        )
                    )
                    policy.population_i = population_i
                    logger.info("{} succeed. A result of load_recent_prob={} with principle={}".format(
                        load_message,
                        load_recent_prob,
                        principle,
                    ))
                except Exception as e:
                    logger.warning("{} failed: {}.".format(
                        load_message,
                        e,
                    ))

            else:
                # if population_i is not in possible_populations yet
                logger.info("{} skipped, the population_i does not exist yet, creating branch (possible_populations are {})".format(
                    load_message,
                    possible_populations,
                ))
                policy.population_i = population_i


def preprocess_config_value_this_level(running_config, config_key_this_level, config_value_this_level, default):

    config_value_this_level = get_list_from_gridsearch(config_value_this_level)

    if config_key_this_level not in ["env"]:

        if not is_arena_env(running_config["env"]):

            if not is_list_match(config_value_this_level, default):

                logger.warning(
                    "None-arena env does not support the config of {}. ".format(
                        config_key_this_level,
                    ) +
                    "Overriding it from {} to [{}].".format(
                        config_value_this_level,
                        default,
                    )
                )
                config_value_this_level = [default]

    return config_value_this_level


def get_env_infos(env, env_config):
    """Create dummy_env to get env_infos of env as a dict.

    Arguments:
        env: id of env
        env_config: env_config
    Returns:
        env_infos
    """

    env_infos = {}

    if is_arena_env(env):
        dummy_env = ArenaRllibEnv(
            env=env,
            env_config=env_config,
        )
        env_infos["number_agents"] = dcopy(
            dummy_env.number_agents
        )
    else:
        dummy_env = gym.make(env)
        env_infos["number_agents"] = 1

    env_infos["obs_space"] = dcopy(
        dummy_env.observation_space
    )
    env_infos["act_space"] = dcopy(
        dummy_env.action_space
    )

    dummy_env.close()

    return env_infos


def expand_exp(config_to_expand, config_keys_to_expand, args=None, parser=None, expanded_exp_key_prefix="", expanded_exps={}, running_config={}):
    """Expand config_to_expand at config_keys_to_expand, where the config_to_expand could be a grid_search.

    Arguments:
        config_to_expand: config to expand
        config_keys_to_expand: keys at which config_to_expand will be expanded
        parser: this is used to get default value of configs
        expanded_exp_key_prefix: prefix of expanded_exp_key
        expanded_exps: holding the expanded_exps which is the final return of the function
    """

    if len(config_keys_to_expand) == 0:

        # if there is not any config_keys_to_expand

        # create expanded_exp_key, the key of expanded_exps
        expanded_exp_key = to_dir_str(
            expanded_exp_key_prefix +
            running_config_to_str(
                running_config
            )
        )

        # create expanded_exps[expanded_exp_key]
        expanded_exps[expanded_exp_key] = dcopy(
            config_to_expand
        )

        # refer expanded_exps[expanded_exp_key] as expanded_exp
        expanded_exp = expanded_exps[expanded_exp_key]

        # update expanded_exp with running_config
        update_config_value_by_config(
            config_to_update=expanded_exp,
            config=running_config,
        )

        if is_arena_env(expanded_exp["env"]):

            # if is arena env

            # update expanded_exp["config"] with infos of env
            expanded_exp["config"].update(
                get_env_infos(
                    env=expanded_exp["env"],
                    env_config=expanded_exp["config"]["env_config"],
                )
            )

            # process expanded_exp["config"]["num_learning_policies"]
            if isinstance(expanded_exp["config"]["num_learning_policies"], str):
                if expanded_exp["config"]["num_learning_policies"] in ["all"]:
                    expanded_exp["config"]["num_learning_policies"] = dcopy(
                        expanded_exp["config"]["number_agents"]
                    )
                    logger.warning(
                        "Override config.num_learning_policies from all to {}".format(
                            expanded_exp["config"]["num_learning_policies"],
                        ) +
                        "This may cause repeated experiments."
                    )
                else:
                    raise NotImplementedError

            # process expanded_exp["config"]["share_layer_policies"]
            if isinstance(expanded_exp["config"]["share_layer_policies"], str):
                if expanded_exp["config"]["share_layer_policies"] in ["team"]:
                    expanded_exp["config"]["share_layer_policies"] = dcopy(
                        get_social_config(
                            expanded_exp["env"]
                        )
                    )
                    logger.warning(
                        "Override config.share_layer_policies from team to {}".format(
                            expanded_exp["config"]["share_layer_policies"],
                        ) +
                        "This may cause repeated experiments."
                    )

            # process expanded_exp["config"]["actor_critic_obs"]
            if not((len(expanded_exp["config"]["actor_critic_obs"]) == 0) or (len(expanded_exp["config"]["actor_critic_obs"]) == 2)):
                raise Exception(
                    "actor_critic_obs can only be [] or [xx, yy]"
                )

            # append necessary configs to expanded_exp["config"]
            expanded_exp["config"].update(
                {
                    # a list of policy ids of which the policy is trained
                    "learning_policy_ids": [],
                    # a list of policy ids of which the policy is not trained
                    "playing_policy_ids": [],
                    "multiagent": {
                        # configs of policies
                        "policies": {},
                        # mapping agent to policy
                        "policy_mapping_fn": ray.tune.function(
                            policy_mapping_fn_i2i
                        )
                    },
                    "callbacks": {
                        # called after each train iteration
                        "on_train_result": ray.tune.function(
                            on_train_result
                        )
                    }
                }
            )

            # create configs of learning policies
            for learning_policy_i in range(expanded_exp["config"]["num_learning_policies"]):
                learning_policy_id = policy_i2id(
                    learning_policy_i
                )
                expanded_exp["config"]["learning_policy_ids"] += [
                    learning_policy_id
                ]

            # create configs of playing policies
            for playing_policy_i in range(expanded_exp["config"]["num_learning_policies"], expanded_exp["config"]["number_agents"]):
                playing_policy_id = policy_i2id(
                    playing_policy_i
                )
                expanded_exp["config"]["playing_policy_ids"] += [
                    playing_policy_id
                ]

            # config.multiagent.policies_to_train is config.learning_policy_ids
            expanded_exp["config"]["multiagent"]["policies_to_train"] = dcopy(
                expanded_exp["config"]["learning_policy_ids"]
            )

            if len(expanded_exp["config"]["playing_policy_ids"]) > 0:

                # if there are playing policy

                if expanded_exp["run"] not in ["PPO"]:
                    # build custom_action_dist to be playing mode dist (no exploration)
                    # TODO: support pytorch policy and other algorithms, currently only add support for tf_action_dist on PPO
                    # see this issue for a fix: https://github.com/ray-project/ray/issues/5729
                    raise NotImplementedError

                if not args.eval:
                    if expanded_exp["config"]["env_config"]["is_shuffle_agents"] == False:
                        logger.warning(
                            "There are playing policies, which keeps loading learning policies. " +
                            "This means you need to shuffle agents so that the learning policies can generalize to playing policies. "
                        )
                        input("# WARNING: Need comfirmation.")

            elif len(expanded_exp["config"]["playing_policy_ids"]) == 0:

                if expanded_exp["config"]["playing_policy_load_recent_prob"] != None:
                    logger.warning(
                        "There are no playing agents." +
                        "Thus, config.playing_policy_load_recent_prob will not taking effect."
                    )
                    input("# WARNING: Need comfirmation.")

            else:
                raise ValueError

            # apply configs of all policies
            for policy_i in range(expanded_exp["config"]["number_agents"]):

                policy_id = policy_i2id(policy_i)

                policy_config = {}

                policy_config["vf_share_layers"] = True,

                if policy_id in expanded_exp["config"]["playing_policy_ids"]:

                    # for playing policy, create custom_action_dist that does not explore
                    from ray.rllib.models.tf.tf_action_dist import Deterministic as DeterministiContinuous
                    policy_config["custom_action_dist"] = {
                        "Discrete": DeterministicCategorical,
                        "Box": DeterministiContinuous
                    }[
                        expanded_exp["config"]["act_space"].__class__.__name__
                    ]

                # config.share_layer_policies and config.actor_critic_obs are implemented in ArenaPolicy
                if (expanded_exp["config"]["share_layer_policies"] != []) or (expanded_exp["config"]["actor_critic_obs"] != []):

                    policy_config["model"] = {}
                    policy_config["model"]["custom_model"] = "ArenaPolicy"

                    if expanded_exp["config"]["share_layer_policies"] != []:

                        # pass custom_options for share_layer_policies
                        policy_config["model"]["custom_options"] = {
                            "shared_scope": get_shared_scope(
                                share_layer_policies=expanded_exp["config"]["share_layer_policies"],
                                policy_i=policy_im
                            ),
                        }

                if expanded_exp["run"] not in ["PPO"]:
                    # # TODO: currently only support PPO
                    raise NotImplementedError
                else:
                    from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

                # finish expanded_exp["config"]["multiagent"]["policies"]
                expanded_exp["config"]["multiagent"]["policies"][policy_id] = (
                    PPOTFPolicy,
                    expanded_exp["config"]["obs_space"],
                    expanded_exp["config"]["act_space"],
                    dcopy(policy_config),
                )

        return expanded_exps

    else:

        config_key_this_level = config_keys_to_expand[0]

        config_value_this_level = get_config_value_by_key(
            config_to_get=dcopy(config_to_expand),
            config_key=config_key_this_level,
        )

        config_value_this_level = preprocess_config_value_this_level(
            running_config=running_config,
            config_key_this_level=config_key_this_level,
            config_value_this_level=config_value_this_level,
            default=parser.get_default(
                get_key_in_parse_from_config_key(
                    config_key_this_level
                )
            ),
        )

        for each_config_value_this_level in config_value_this_level:
            running_config[config_key_this_level] = each_config_value_this_level
            expanded_exps = expand_exp(
                config_to_expand=dcopy(config_to_expand),
                config_keys_to_expand=config_keys_to_expand[1:],
                args=args,
                parser=parser,
                expanded_exp_key_prefix=expanded_exp_key_prefix,
                expanded_exps=expanded_exps,
                running_config=running_config,
            )

        return expanded_exps


def create_arena_exps(exps, args, parser):
    """Create arena_exps from exps
    Expand exps with grid_search, this is implemented to override the default support of grid_search for customized configs.
    """

    if args.eval:
        exps = override_exps_to_eval(exps)

    if args.dummy:
        exps = override_exps_to_dummy(exps)

    arena_exps = {}

    for exp_key, config in exps.items():

        if args.eager:
            config["config"]["eager"] = True

        arena_exps.update(
            expand_exp(
                config_to_expand=dcopy(config),
                config_keys_to_expand=[
                    "env",
                    "config-num_learning_policies",
                    "config-share_layer_policies",
                    "config-actor_critic_obs",
                    "config-env_config-sensors",
                    "config-env_config-multi_agent_obs",
                    "config-env_config-is_shuffle_agents",
                ],
                args=args,
                parser=parser,
                expanded_exp_key_prefix=exp_key,
            )
        )

    return arena_exps
