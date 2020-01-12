import logging
import ray
import platform
import random

from copy import deepcopy as dcopy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym_unity.envs import UnityEnv

from .utils import *
from .constants import *

logger = logging.getLogger(__name__)


class ArenaRllibEnv(MultiAgentEnv):
    """Convert ArenaUnityEnv(gym_unity) to MultiAgentEnv (rllib)
    """

    # Arena auto reset on the BuildingToolkit side, so keeo this to True
    IS_AUTO_RESET = True

    # this need to be compatible with Arena-BuildingToolkit
    VISUAL_OBS_INDEX = {
        "visual_FP": 0,
        "visual_TP": 1,
    }

    def __init__(self, env, env_config):

        self.env = env
        if self.env is None:
            error = "Config env in has to be specified"
            logger.error(error)
            raise Exception(error)

        self.obs_type = env_config.get(
            "obs_type", "visual_FP")
        self.obs_type = self.obs_type.split("-")

        if (self.obs_type[0] == "vector") and (len(self.obs_type) == 1):
            self.use_visual = False
        else:
            if "vector" not in self.obs_type:
                self.use_visual = True
            else:
                error = "Combining visual and vector obs (self.obs_type={}) is not supported.".format(
                    self.obs_type
                )
                logger.error(error)
                raise Exception(error)

        self.is_multi_agent_obs = env_config.get("is_multi_agent_obs", False)

        game_file_path, extension_name = get_env_directory(self.env)

        # check of we can use a server build
        if self.obs_type == "vector":
            if os.path.exists(game_file_path + '-Server'):
                game_file_path = game_file_path + '-Server'
                logger.info(
                    "Using server build."
                )
            else:
                logger.warning(
                    "Only vector observation is used, you can have a server build which runs faster."
                )
        else:
            logger.info(
                "Using full build."
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
                rank = random.randint(0, 65535)
                self.env = ArenaUnityEnv(
                    game_file_path,
                    rank,
                    use_visual=self.use_visual,
                    uint8_visual=False,
                    multiagent=True,
                    allow_multiple_visual_obs=True,
                )
                break
            except Exception as e:
                logger.warning("Start ArenaUnityEnv failed {}, retrying...".format(
                    e
                ))

        self.env.set_train_mode(train_mode=env_config.get("train_mode", True))

        self.is_shuffle_agents = env_config.get("is_shuffle_agents", False)
        if self.is_shuffle_agents:
            self.shift = 0

        self.action_space = self.env.action_space

        own_observation_space = dcopy(
            self.env.observation_space
        )

        if self.use_visual:

            num_visual_obs = 0
            for obs_type in self.obs_type:
                if "visual" in obs_type:
                    num_visual_obs += 1

            own_observation_space.shape = replace_in_tuple(
                tup=own_observation_space.shape,
                index=2,
                value=num_visual_obs,
            )

            self.visual_obs_indices = []
            for obs_type_ in self.obs_type:
                self.visual_obs_indices += [
                    ArenaRllibEnv.VISUAL_OBS_INDEX[obs_type_]
                ]

        if self.is_multi_agent_obs:
            self.observation_space = {}
            self.observation_space[
                "own_obs"
            ] = own_observation_space
            self.observation_space = gym.spaces.Dict(
                self.observation_space
            )
        else:
            self.observation_space = own_observation_space

        self.number_agents = self.env.number_agents

        # # the first episode has bug, not sure what causes this
        # for i in range(100):
        #     self.reset()

    def run_an_episode(self, actions=None):
        """Run an episode with actions at each step.
        If actions is not provided, use 0 as actions
        """
        self.reset()

        if actions is None:
            actions = {}
            for agent_i in range(self.number_agents):
                actions[self.get_agent_id(agent_i)] = 0

        while True:
            _, _, dones, _ = self.step(actions)
            if dones["__all__"]:
                break

    def process_agent_obs(self, agent_obs):
        """Take and concatenate multiple obs at dimension 0 to the channel dimension
        """
        if self.use_visual:
            return np.concatenate(
                np.take(
                    agent_obs,
                    indices=self.visual_obs_indices,
                    axis=0,
                ),
                axis=2,
            )
        else:
            return agent_obs

    def process_obs(self, obs_gym_unity):

        obs = {}
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            if self.is_multi_agent_obs:
                obs[agent_id] = {}
                obs[agent_id]["own_obs"] = self.process_agent_obs(
                    obs_gym_unity[agent_i]
                )
            else:
                obs[agent_id] = self.process_agent_obs(
                    obs_gym_unity[agent_i]
                )

        return obs

    def process_returns(self, obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity):

        obs_rllib = self.process_obs(obs_gym_unity)

        rewards_rllib = {}
        dones_rllib = {}
        infos_rllib = {}
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            rewards_rllib[agent_id] = rewards_gym_unity[agent_i]
            dones_rllib[agent_id] = dones_gym_unity[agent_i]
            infos_rllib[agent_id] = infos_gym_unity

        # done when all agents are done
        dones_rllib["__all__"] = np.all(dones_gym_unity)

        return obs_rllib, rewards_rllib, dones_rllib, infos_rllib

    def process_actions(self, actions_rllib):
        actions_gym_unity = []
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            actions_gym_unity += [actions_rllib[agent_id]]
        return actions_gym_unity

    def shuffle_actions_gym_unity(self, actions_gym_unity):
        return self.roll(actions_gym_unity).tolist()

    def shuffle_returns_gym_unity(self, obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity):

        rewards_gym_unity = self.roll_back(rewards_gym_unity)
        dones_gym_unity = self.roll_back(dones_gym_unity)
        infos_gym_unity["shift"] = self.shift
        return obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity

    def shuffle_obs_gym_unity(self, obs_gym_unity):
        obs_gym_unity = self.roll_back(obs_gym_unity)
        return obs_gym_unity

    def reset(self):

        if self.is_shuffle_agents:
            self.shift = np.random.randint(0, self.number_agents)

        obs_gym_unity = self.env.reset()

        # shuffle (gym_unity)
        if self.is_shuffle_agents:
            obs_gym_unity = self.shuffle_obs_gym_unity(obs_gym_unity)

        # returns_gym_unity to returns_rllib
        obs_rllib = self.process_obs(obs_gym_unity)

        return obs_rllib

    def step(self, actions_rllib):

        # actions_rllib to actions_gym_unity
        actions_gym_unity = self.process_actions(actions_rllib)

        # shuffle (gym_unity)
        if self.is_shuffle_agents:
            actions_gym_unity = self.shuffle_actions_gym_unity(
                actions_gym_unity
            )

        # step forward (gym_unity)
        obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity = self.env.step(
            actions_gym_unity
        )

        # shuffle (gym_unity)
        if self.is_shuffle_agents:
            obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity = self.shuffle_returns_gym_unity(
                obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity
            )

        # returns_gym_unity to returns_rllib
        obs_rllib, rewards_rllib, dones_rllib, infos_rllib = self.process_returns(
            obs_gym_unity, rewards_gym_unity, dones_gym_unity, infos_gym_unity
        )

        # done (rllib)
        if dones_rllib["__all__"] and ArenaRllibEnv.IS_AUTO_RESET:
            obs_rllib = self.reset()

        return obs_rllib, rewards_rllib, dones_rllib, infos_rllib

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
        """arena-spec: ml-agent does not support multiple agents & multiple obs
        """

        multiple_visual_obs = np.asarray(multiple_visual_obs)
        # (multiple visual obs, multiple agents, 84, 84, 1)

        multiple_visual_obs = np.transpose(
            multiple_visual_obs,
            axes=(1, 0, 2, 3, 4),
        )
        # (multiple agents, multiple visual obs, 84, 84, 1)

        if self.uint8_visual:
            multiple_visual_obs = (
                255.0 * multiple_visual_obs
            ).astype(np.uint8)

        return list(multiple_visual_obs)

    # arena-spec
    def set_train_mode(self, train_mode):
        self.train_mode = train_mode

    def reset(self):
        """arena-spec: add support for train_mode=self.train_mode
        """
        info = self._env.reset(train_mode=self.train_mode)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self.game_over = False

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
        else:
            obs, reward, done, info = self._multi_step(info)
        return obs


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
