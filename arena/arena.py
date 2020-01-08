import os
import platform
import random
import time
import logging
import copy
import glob
import pickle

import numpy as np

from gym_unity.envs import UnityEnv

import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.tf.tf_action_dist import Deterministic as DeterministiContinuous

from .models import DeterministicCategorical
from .utils import get_list_from_gridsearch, get_one_from_grid_search, is_grid_search
from .utils import prepare_path
from .utils import list_subtract, find_in_list_of_list

ARENA_ENV_PREFIX = 'Arena-'
AGENT_ID_PREFIX = "agent"
POLICY_ID_PREFIX = "policy"

CHECKPOINT_PATH_PREFIX = "learning_agent/"
CHECKPOINT_PATH_POPULATION_PREFIX = "p_"
CHECKPOINT_PATH_ITERATION_PREFIX = "i_"

logger = logging.getLogger(__name__)


class ArenaRllibEnv(MultiAgentEnv):
    """Convert ArenaUnityEnv(gym_unity) to MultiAgentEnv (rllib)
    """

    def __init__(self, env, env_config):

        self.env = env
        if self.env is None:
            raise Exception("env in has to be specified")

        self.obs_type = env_config.get("obs_type", "visual_FP")

        if "-" in self.obs_type:
            input('# TODO: multiple obs support')

        if self.obs_type in ["visual_TP"]:
            input('# TODO: visual_TP obs support')

        game_file_path, extension_name = get_env_directory(self.env)

        if self.obs_type in ["vector"]:

            if os.path.exists(game_file_path + '-Server'):
                game_file_path = game_file_path + '-Server'
                logger.info(
                    "Using server build"
                )

            else:
                logger.warning(
                    "Only vector observation is used, you can have a server build which runs faster"
                )

        else:

            logger.info(
                "Using full build"
            )

        if not os.path.exists(game_file_path + extension_name):

            error = "Game build {} does not exist".format(
                game_file_path
            )
            logger.error(error)
            raise Exception(error)

        while True:
            try:
                # TODO: Individual game instance cannot get rank from rllib, so just try ranks
                rank = random.randint(0, 65534)
                self.env = ArenaUnityEnv(
                    game_file_path,
                    rank,
                    use_visual=False if self.obs_type in ["vector"] else True,
                    uint8_visual=False,
                    multiagent=True,
                )
                break
            except Exception as e:
                pass

        self.env.set_train_mode(train_mode=env_config.get("train_mode", True))

        self.is_shuffle_agents = env_config.get("is_shuffle_agents", False)
        if self.is_shuffle_agents:
            self.shift = 0

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.number_agents = self.env.number_agents

    def reset(self):

        if self.is_shuffle_agents:
            self.shift = np.random.randint(0, self.number_agents)

        obs_ = self.env.reset()

        if self.is_shuffle_agents:
            obs_ = self.roll_back(obs_)

        # xxx_ (gym_unity) to xxx (rllib)
        obs = {}
        for agent_i in range(self.number_agents):
            obs[self.get_agent_id(agent_i)] = obs_[agent_i]

        return obs

    def step(self, actions):

        # xxx (rllib) to xxx_ (gym_unity)
        actions_ = []
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            actions_ += [actions[agent_id]]

        if self.is_shuffle_agents:
            actions_ = self.roll(actions_).tolist()

        # step forward (gym_unity)
        obs_, rewards_, dones_, infos_ = self.env.step(actions_)

        if self.is_shuffle_agents:
            obs_ = self.roll_back(obs_)
            rewards_ = self.roll_back(rewards_)
            dones_ = self.roll_back(dones_)
            infos_["shift"] = self.shift

        # xxx_ (gym_unity) to xxx (rllib)
        obs = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            obs[agent_id] = obs_[agent_i]
            rewards[agent_id] = rewards_[agent_i]
            dones[agent_id] = dones_[agent_i]
            infos[agent_id] = infos_

        # done when all agents are done
        dones["__all__"] = np.all(dones_)

        return obs, rewards, dones, infos

    def roll(self, x):
        return np.roll(x, self.shift, axis=0)

    def roll_back(self, x):
        return np.roll(x, -self.shift, axis=0)

    def close(self):
        self.env.close()

    def get_agent_ids(self):
        self.agent_ids = []
        for agent_i in range(self.number_agents):
            self.agent_ids += [self.get_agent_id(agent_i)]
        return self.agent_ids

    def get_agent_id(self, agent_i):
        return "{}_{}".format(AGENT_ID_PREFIX, agent_i)

    def get_agent_i(self, agent_id):
        return int(agent_id.split(AGENT_ID_PREFIX + "_")[1])

    def get_agent_id_prefix(self):
        return AGENT_ID_PREFIX


class ArenaUnityEnv(UnityEnv):
    """An override of UnityEnv from gym_unity.envs, to fix some of their bugs and add some supports.
    Search "arena-spec" for these places.
    """

    def _preprocess_multi(self, multiple_visual_obs):
        if self.uint8_visual:
            return [
                (255.0 * _visual_obs).astype(np.uint8)
                # arena-spec: change multiple_visual_obs to multiple_visual_obs[0], this is an ml-agent bug
                for _visual_obs in multiple_visual_obs[0]
            ]
        else:
            # arena-spec: change multiple_visual_obs to multiple_visual_obs[0], this is an ml-agent bug
            return multiple_visual_obs[0]

    # arena-spec
    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        # arena-spec: add train_mode=self.train_mode
        info = self._env.reset(train_mode=self.train_mode)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self.game_over = False

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
        else:
            obs, reward, done, info = self._multi_step(info)
        return obs


def get_policy_id(policy_i):
    """Get policy_id from policy_i.
    """
    return "{}_{}".format(POLICY_ID_PREFIX, policy_i)


def get_agent_i(agent_id):
    """Get agent_i from agent_id.
    """
    return int(agent_id.split(AGENT_ID_PREFIX + "_")[1])


def policy_mapping_fn_i2i(agent_id):
    """A policy_mapping_fn that maps agent i to policy i.
    """
    return get_policy_id(get_agent_i(agent_id))


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


def remove_arena_env_prefix(env):
    """Remove ARENA_ENV_PREFIX from a env (possibly a grid_search).
    """
    env = copy.deepcopy(env)
    if is_grid_search(env):
        if is_all_arena_env(env):
            for i in range(len(env["grid_search"])):
                env["grid_search"][i] = remove_arena_env_prefix(
                    env["grid_search"][i]
                )
            return env
        else:
            raise NotImplementedError
    else:
        return env[len(ARENA_ENV_PREFIX):]


def get_env_directory(env_name):
    """Get env path according to env_name
    """
    return os.path.join(
        os.path.dirname(__file__),
        "bin/{}-{}".format(
            env_name,
            platform.system(),
        )
    ), {
        "Linux": ".x86_64",
        "Darwin": ".app",
    }[platform.system()]


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
                logger.warning("{} failed: {}".format(
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
                    logger.warning("{} failed: {}".format(
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


def create_arena_experiments(experiments, args, parser):
    """Create arena experiments from experiments
    Expand experiments with grid_search, this is implemented to override the default support of grid_search for customized configs
    """
    arena_experiments = {}
    for experiment_key in experiments.keys():

        env_keys = get_list_from_gridsearch(
            experiments[experiment_key]["env"]
        )

        for env in env_keys:

            # check config, if invalid, skip
            if not is_arena_env(env):
                logger.warn("env {} is not arena env, skipping".format(
                    env
                ))
                continue

            # accept config
            experiments[experiment_key]["env"] = copy.deepcopy(
                env
            )

            # create dummy_env to get parameters/setting of env
            dummy_env = ArenaRllibEnv(
                env=remove_arena_env_prefix(
                    experiments[experiment_key]["env"]
                ),
                env_config=experiments[experiment_key]["config"]["env_config"],
            )

            experiments[experiment_key]["config"]["num_agents"] = copy.deepcopy(
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

            num_learning_policies_keys = get_list_from_gridsearch(
                experiments[experiment_key]["config"]["num_learning_policies"]
            )

            for num_learning_policies in num_learning_policies_keys:

                # get other keys
                num_learning_policies_other_keys = copy.deepcopy(
                    num_learning_policies_keys
                )
                num_learning_policies_other_keys.remove(
                    num_learning_policies
                )

                # override config
                if num_learning_policies == "all":
                    num_learning_policies = int(
                        experiments[experiment_key]["config"]["num_agents"]
                    )

                # if in other keys ofter override, skip this config
                if num_learning_policies in num_learning_policies_other_keys:
                    continue

                # accept the config
                experiments[experiment_key]["config"]["num_learning_policies"] = copy.deepcopy(
                    num_learning_policies
                )

                share_layer_policies_keys = get_list_from_gridsearch(
                    experiments[experiment_key]["config"]["share_layer_policies"]
                )

                for share_layer_policies in share_layer_policies_keys:

                    # get other keys
                    share_layer_policies_other_keys = copy.deepcopy(
                        share_layer_policies_keys
                    )
                    share_layer_policies_other_keys.remove(
                        share_layer_policies
                    )

                    # override config
                    if isinstance(share_layer_policies, list):
                        if np.max(np.asarray(share_layer_policies)) >= experiments[experiment_key]["config"]["num_agents"]:
                            logger.warning(
                                "There are policy_i that exceeds num_agents in config.share_layer_policies. Disable config.share_layer_policies."
                            )
                            share_layer_policies = "none"

                    elif isinstance(share_layer_policies, str):
                        if share_layer_policies == "team":
                            logger.warning(
                                "# TODO: not supported yet. Override config.share_layer_policies to none."
                            )
                            share_layer_policies = "none"
                        elif share_layer_policies == "none":
                            pass
                        else:
                            raise NotImplementedError

                    else:
                        raise NotImplementedError

                    # if in other keys ofter override, skip this config
                    if share_layer_policies in share_layer_policies_other_keys:
                        continue

                    # accept the config
                    experiments[experiment_key]["config"]["share_layer_policies"] = copy.deepcopy(
                        share_layer_policies
                    )

                    grid_experiment_key = "{}_num_learning_policies={},share_layer_policies={}".format(
                        experiment_key,
                        experiments[experiment_key]["config"]["num_learning_policies"],
                        experiments[experiment_key]["config"]["share_layer_policies"],
                    )

                    arena_experiments[grid_experiment_key] = copy.deepcopy(
                        experiments[experiment_key]
                    )

                    if not arena_experiments[grid_experiment_key].get("run"):
                        parser.error(
                            "The following arguments are required: --run"
                        )
                    if not arena_experiments[grid_experiment_key].get("env") and not arena_experiments[grid_experiment_key].get("config", {}).get("env"):
                        parser.error(
                            "The following arguments are required: --env"
                        )
                    if args.eager:
                        arena_experiments[grid_experiment_key]["config"]["eager"] = True

                    # policies: config of policies
                    arena_experiments[grid_experiment_key]["config"]["multiagent"] = {
                    }
                    arena_experiments[grid_experiment_key]["config"]["multiagent"]["policies"] = {
                    }
                    # learning_policy_ids: a list of policy ids of which the policy is trained
                    arena_experiments[grid_experiment_key]["config"]["learning_policy_ids"] = [
                    ]
                    # playing_policy_ids: a list of policy ids of which the policy is not trained
                    arena_experiments[grid_experiment_key]["config"]["playing_policy_ids"] = [
                    ]

                    # create configs of learning policies
                    for learning_policy_i in range(arena_experiments[grid_experiment_key]["config"]["num_learning_policies"]):
                        learning_policy_id = get_policy_id(
                            learning_policy_i
                        )
                        arena_experiments[grid_experiment_key]["config"]["learning_policy_ids"] += [
                            learning_policy_id
                        ]

                    if arena_experiments[grid_experiment_key]["run"] not in ["PPO"]:
                        # build custom_action_dist to be playing mode dist (no exploration)
                        # TODO: support pytorch policy and other algorithms, currently only add support for tf_action_dist on PPO
                        # see this issue for a fix: https://github.com/ray-project/ray/issues/5729
                        raise NotImplementedError

                    # create configs of playing policies
                    for playing_policy_i in range(arena_experiments[grid_experiment_key]["config"]["num_learning_policies"], experiments[experiment_key]["config"]["num_agents"]):
                        playing_policy_id = get_policy_id(
                            playing_policy_i
                        )
                        arena_experiments[grid_experiment_key]["config"]["playing_policy_ids"] += [
                            playing_policy_id
                        ]

                    if len(arena_experiments[grid_experiment_key]["config"]["playing_policy_ids"]) > 0:

                        if arena_experiments[grid_experiment_key]["config"]["env_config"]["is_shuffle_agents"] == False:
                            logger.warning(
                                "There are playing policies, which keeps loading learning policies. This means you need to shuffle agents so that the learning policies can generalize to playing policies. Overriding config.env_config.is_shuffle_agents to True."
                            )
                            arena_experiments[grid_experiment_key]["config"]["env_config"]["is_shuffle_agents"] = True

                    elif len(arena_experiments[grid_experiment_key]["config"]["playing_policy_ids"]) == 0:

                        if arena_experiments[grid_experiment_key]["config"]["playing_policy_load_recent_prob"] != "none":
                            logger.warning(
                                "There are no playing agents. Thus, config.playing_policy_load_recent_prob is invalid. Overriding it to none."
                            )
                            arena_experiments[grid_experiment_key]["config"]["playing_policy_load_recent_prob"] = "none"

                    else:
                        raise ValueError

                    # apply configs of all policies
                    for policy_i in range(experiments[experiment_key]["config"]["num_agents"]):

                        policy_id = get_policy_id(policy_i)

                        policy_config = {}

                        if policy_id in arena_experiments[grid_experiment_key]["config"]["playing_policy_ids"]:
                            policy_config["custom_action_dist"] = {
                                "Discrete": DeterministicCategorical,
                                "Box": DeterministiContinuous
                            }[
                                arena_experiments[grid_experiment_key]["config"]["act_space"].__class__.__name__
                            ]

                        policy_config["model"] = {}

                        if experiments[experiment_key]["config"]["share_layer_policies"] != "none":
                            policy_config["model"]["custom_model"] = "ShareLayerPolicy"
                            policy_config["model"]["custom_options"] = {}
                            policy_config["model"]["custom_options"]["shared_scope"] = copy.deepcopy(
                                experiments[experiment_key]["config"]["share_layer_policies"][
                                    find_in_list_of_list(
                                        experiments[experiment_key]["config"]["share_layer_policies"], policy_i
                                    )[0]
                                ]
                            )

                        arena_experiments[grid_experiment_key]["config"]["multiagent"]["policies"][policy_id] = (
                            None,
                            arena_experiments[grid_experiment_key]["config"]["obs_space"],
                            arena_experiments[grid_experiment_key]["config"]["act_space"],
                            copy.deepcopy(policy_config),
                        )

                    # policy_mapping_fn: a map from agent_id to policy_id
                    # use policy_mapping_fn_i2i as policy_mapping_fn
                    arena_experiments[grid_experiment_key]["config"]["multiagent"]["policy_mapping_fn"] = ray.tune.function(
                        policy_mapping_fn_i2i
                    )

                    arena_experiments[grid_experiment_key]["config"]["callbacks"] = {
                    }
                    arena_experiments[grid_experiment_key]["config"]["callbacks"]["on_train_result"] = ray.tune.function(
                        on_train_result
                    )

                    arena_experiments[grid_experiment_key]["config"]["multiagent"]["policies_to_train"] = copy.deepcopy(
                        arena_experiments[grid_experiment_key]["config"]["learning_policy_ids"]
                    )

    return arena_experiments
