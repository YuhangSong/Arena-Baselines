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
    return possible_populations


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
    return possible_iterations


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


def check_config_keys(env, config_keys, default):
    if isinstance(config_keys, list):
        if not is_arena_env(env):
            if is_gridsearch_match(config_keys, default):
                logger.warning(
                    "None-arena env does not support the config of {}. ".format(
                        config_keys,
                    ) +
                    "Overriding it to {}.".format(
                        default,
                    )
                )
                config_keys = default
    else:
        raise TypeError
    return config_keys


def get_and_check_config_keys_for_arena(env, config, default):
    return check_config_keys(
        env=env,
        config_keys=get_list_from_gridsearch(
            config=config,
        ),
        default=default,
    )


def get_env_infos(env, env_config):
    """Create dummy_env to get parameters/setting of env
    """

    if is_arena_env(env):
        dummy_env = ArenaRllibEnv(
            env=env,
            env_config=env_config,
        )
        number_agents = dcopy(
            dummy_env.number_agents
        )
    else:
        dummy_env = gym.make(env)
        number_agents = 1
    obs_space = dcopy(
        dummy_env.observation_space
    )
    act_space = dcopy(
        dummy_env.action_space
    )
    dummy_env.close()

    return number_agents, obs_space, act_space


def expand_exp(exp_to_expand, config_keys_to_expand, args, parser, expanded_exp_key_prefix="", running_configs=None, expanded_exps=None):

    if running_configs is None:
        # create running_configs, this will hold the swapping configs
        running_configs = {}

    if expanded_exps is None:
        # create expanded_exps, which is the final return of the function
        expanded_exps = {}

    if len(config_keys_to_expand) == 0:

        expanded_exp_key = expanded_exp_key_prefix

        # represent running_configs in expanded_exp_key
        for running_config_key, running_config in running_configs.items():
            expanded_exp_key += ",{}={}".format(
                simplify_config_key(str(running_config_key)),
                running_config,
            )

        expanded_exp_key = to_dir_str(expanded_exp_key)

        expanded_exps[expanded_exp_key] = dcopy(
            exp_to_expand
        )

        expanded_exp = expanded_exps[expanded_exp_key]

        # assign running_configs to expanded_exp
        for running_config_key, running_config in running_configs.items():
            temp = expanded_exp
            running_config_key = running_config_key.split("-")
            len_running_config_key = len(running_config_key)
            for i in range(len_running_config_key):
                if i < (len_running_config_key - 1):
                    temp = temp[running_config_key[i]]
                elif i == (len_running_config_key - 1):
                    temp[running_config_key[i]] = running_config
                else:
                    raise ValueError

        if is_arena_env(expanded_exp["env"]):

            number_agents, obs_space, act_space = get_env_infos(
                env=expanded_exp["env"],
                env_config=expanded_exp["config"]["env_config"],
            )

            expanded_exp["config"].update(
                {
                    "number_agents": number_agents,
                    "obs_space": obs_space,
                    "act_space": act_space,
                    # a list of policy ids of which the policy is trained
                    "learning_policy_ids": [],
                    # a list of policy ids of which the policy is not trained
                    "playing_policy_ids": [],
                    "multiagent": {
                        # configs of policies
                        "policies": {}
                    },
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

            if expanded_exp["run"] not in ["PPO"]:
                # build custom_action_dist to be playing mode dist (no exploration)
                # TODO: support pytorch policy and other algorithms, currently only add support for tf_action_dist on PPO
                # see this issue for a fix: https://github.com/ray-project/ray/issues/5729
                raise NotImplementedError

            # create configs of playing policies
            for playing_policy_i in range(expanded_exp["config"]["num_learning_policies"], expanded_exp["config"]["number_agents"]):
                playing_policy_id = policy_i2id(
                    playing_policy_i
                )
                expanded_exp["config"]["playing_policy_ids"] += [
                    playing_policy_id
                ]

            if len(expanded_exp["config"]["playing_policy_ids"]) > 0:

                if expanded_exp["config"]["env_config"]["is_shuffle_agents"] == False:
                    logger.warning(
                        "There are playing policies, which keeps loading learning policies. This means you need to shuffle agents so that the learning policies can generalize to playing policies. Overriding config.env_config.is_shuffle_agents to True."
                    )
                    expanded_exp["config"]["env_config"]["is_shuffle_agents"] = True

            elif len(expanded_exp["config"]["playing_policy_ids"]) == 0:

                if expanded_exp["config"]["playing_policy_load_recent_prob"] != None:
                    logger.warning(
                        "There are no playing agents. Thus, config.playing_policy_load_recent_prob is invalid. Overriding config.playing_policy_load_recent_prob to None."
                    )
                    expanded_exp["config"]["playing_policy_load_recent_prob"] = None

            else:
                raise ValueError

            # apply configs of all policies
            for policy_i in range(expanded_exp["config"]["number_agents"]):

                policy_id = policy_i2id(policy_i)

                policy_config = {}

                if policy_id in expanded_exp["config"]["playing_policy_ids"]:
                    from ray.rllib.models.tf.tf_action_dist import Deterministic as DeterministiContinuous
                    policy_config["custom_action_dist"] = {
                        "Discrete": DeterministicCategorical,
                        "Box": DeterministiContinuous
                    }[
                        expanded_exp["config"]["act_space"].__class__.__name__
                    ]

                policy_config["model"] = {}

                if expanded_exp["config"]["share_layer_policies"] != []:
                    policy_config["model"]["custom_model"] = "ArenaPolicy"

                if expanded_exp["config"]["share_layer_policies"] != []:
                    policy_config["model"]["custom_options"] = {
                    }
                    policy_config["model"]["custom_options"]["shared_scope"] = dcopy(
                        expanded_exp["config"]["share_layer_policies"][
                            find_in_list_of_list(
                                expanded_exp["config"]["share_layer_policies"], policy_i
                            )[0]
                        ]
                    )

                expanded_exp["config"]["multiagent"]["policies"][policy_id] = (
                    None,
                    expanded_exp["config"]["obs_space"],
                    expanded_exp["config"]["act_space"],
                    dcopy(policy_config),
                )

            # policy_mapping_fn: a map from agent_id to policy_id
            # use policy_mapping_fn_i2i as policy_mapping_fn
            expanded_exp["config"]["multiagent"]["policy_mapping_fn"] = ray.tune.function(
                policy_mapping_fn_i2i
            )

            expanded_exp["config"]["callbacks"] = {
                "on_train_result": ray.tune.function(
                    on_train_result
                )
            }
            expanded_exp["config"]["multiagent"]["policies_to_train"] = dcopy(
                expanded_exp["config"]["learning_policy_ids"]
            )

        return expanded_exps

    else:

        # get config_to_expand from exp_to_expand by config_key_to_expand
        config_key_to_expand = config_keys_to_expand[0]
        config_to_expand = exp_to_expand
        config_key_to_expand_in_parse = None
        for config_key_to_expand_each in config_key_to_expand.split("-"):
            config_to_expand = config_to_expand[config_key_to_expand_each]
            # config_key_to_expand_in_parse is the final one in config_key_to_expand.split("-")
            config_key_to_expand_in_parse = config_key_to_expand_each

        if config_key_to_expand in ["env"]:
            config_to_expand_list = get_list_from_gridsearch(
                config_to_expand
            )
        else:
            config_to_expand_list = get_and_check_config_keys_for_arena(
                env=running_configs["env"],
                config=config_to_expand,
                default=parser.get_default(
                    config_key_to_expand_in_parse
                ),
            )

        for config_to_expanded in config_to_expand_list:
            running_configs[config_key_to_expand] = config_to_expanded
            expanded_exps = expand_exp(
                exp_to_expand=exp_to_expand,
                config_keys_to_expand=config_keys_to_expand[1:],
                args=args,
                parser=parser,
                expanded_exp_key_prefix=expanded_exp_key_prefix,
                running_configs=running_configs,
                expanded_exps=expanded_exps,
            )

        return expanded_exps


def create_arena_exps(exps, args, parser):
    """Create arena_exps from exps
    Expand exps with grid_search, this is implemented to override the default support of grid_search for customized configs
    """

    exps = override_exps_according_to_dummy(
        exps=exps,
        dummy=args.dummy,
    )

    arena_exps = {}

    for exp_key, exp in exps.items():

        if args.eager:
            exp["config"]["eager"] = True

        arena_exps.update(
            expand_exp(
                exp_to_expand=exp,
                config_keys_to_expand=[
                    "env",
                    "config-num_learning_policies",
                    "config-share_layer_policies",
                    "config-actor_critic_obs",
                    "config-env_config-is_shuffle_agents",
                    "config-env_config-sensors",
                    "config-env_config-multi_agent_obs",

                ],
                args=args,
                parser=parser,
                expanded_exp_key_prefix=exp_key,
            )
        )

        # share_layer_policies_keys = preprocess_share_layer_policies_keys(
        #     share_layer_policies_keys=share_layer_policies_keys,
        #     env=env,
        # )
        #
        # is_shuffle_agents_keys = preprocess_is_shuffle_agents_keys(
        #     is_shuffle_agents_keys=is_shuffle_agents_keys,
        #     share_layer_policies=share_layer_policies,
        # )
        #
        # varify_actor_critic_obs(actor_critic_obs)
        #
        # multi_agent_obs_keys = preprocess_multi_agent_obs_keys(
        #     multi_agent_obs_keys=multi_agent_obs_keys,
        #     actor_critic_obs=actor_critic_obs,
        # )
        #
        # num_learning_policies_keys = preprocess_num_learning_policies_keys(
        #     num_learning_policies_keys=num_learning_policies_keys,
        #     number_agents=number_agents,
        # )

    return arena_exps
