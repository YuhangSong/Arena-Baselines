from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym_unity.envs import UnityEnv

from .utils import *
from .constants import *

logger = logging.getLogger(__name__)


class ArenaRllibEnv(MultiAgentEnv):
    """Convert ArenaUnityEnv(gym_unity) to MultiAgentEnv (rllib)
    """

    """Following configurations need to be compatible with Arena-BuildingToolkit.
    """
    IS_AUTO_RESET = True
    VISUAL_SENSOR_INDEX = {
        "visual_FP": 0,
        "visual_TP": 1,
    }

    def __init__(self, env, env_config):

        if env is None:
            error = "Config env in has to be specified"
            logger.error(error)
            raise Exception(error)

        self.env_name = env
        self.social_config = get_social_config(self.env_name)

        self.sensors = env_config.get(
            "sensors", ["vector"]
        )

        self.multi_agent_obs = env_config.get(
            "multi_agent_obs", ["own"]
        )

        self.dimension_to_cat_multi_agent_obs_for_sensor = {
            "visual_FP": 2,
            "visual_TP": 2,
            "vector": 0,
        }

        game_file_path, extension_name = get_env_directory(self.env_name)

        # check of we can use a server build
        if is_list_match(self.sensors, "vector"):
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
                game_file_path + extension_name
            )
            logger.error(error)
            raise Exception(error)
        while True:
            try:
                # TODO: Individual game instance cannot get rank from rllib, so just try ranks
                rank = random.randint(10000, 60000)
                self.env = ArenaUnityEnv(
                    game_file_path,
                    rank,
                    use_visual=False,
                    uint8_visual=False,
                    multiagent=True,
                    allow_multiple_visual_obs=True,
                )
                break
            except Exception as e:
                logger.warning("Start ArenaUnityEnv failed {}, retrying...".format(
                    e
                ))

        self.number_agents = dcopy(self.env.number_agents)

        self.train_mode = env_config.get("train_mode", True)
        self.env.set_train_mode(self.train_mode)

        self.is_shuffle_agents = env_config.get("is_shuffle_agents", False)

        self.agent_i_rllib2gymunity = np.arange(self.number_agents)
        self.agent_i_gymunity2rllib = np.arange(self.number_agents)
        self.sync_agent_i_gymunity2rllib()

        self.agent_i_gymunity_mapping = {}

        self.agent_i_gymunity_mapping["own"] = {}
        for agent_i in range(self.number_agents):
            self.agent_i_gymunity_mapping["own"][agent_i] = [agent_i]

        self.agent_i_gymunity_mapping["team_absolute"] = {}
        self.agent_i_gymunity_mapping["team_relative"] = {}
        for agent_i in range(self.number_agents):
            agent_i_social_coordinates = find_in_list_of_list(
                self.social_config, agent_i
            )
            self.agent_i_gymunity_mapping["team_absolute"][agent_i] = dcopy(
                self.social_config[agent_i_social_coordinates[0]]
            )
            self.agent_i_gymunity_mapping["team_relative"][agent_i] = dcopy(
                self.agent_i_gymunity_mapping["team_absolute"][agent_i]
            )
            self.agent_i_gymunity_mapping["team_relative"][agent_i] = list_subtract(
                self.agent_i_gymunity_mapping["team_relative"][agent_i],
                self.agent_i_gymunity_mapping["own"][agent_i]
            )
            self.agent_i_gymunity_mapping["team_relative"][agent_i] += self.agent_i_gymunity_mapping["own"][agent_i]

        self.agent_i_gymunity_mapping["all_absolute"] = {}
        self.agent_i_gymunity_mapping["all_relative"] = {}
        for agent_i in range(self.number_agents):
            self.agent_i_gymunity_mapping["all_absolute"][agent_i] = dcopy(
                flatten_list(self.social_config)
            )
            self.agent_i_gymunity_mapping["all_relative"][agent_i] = dcopy(
                self.agent_i_gymunity_mapping["all_absolute"][agent_i]
            )
            self.agent_i_gymunity_mapping["all_relative"][agent_i] = list_subtract(
                self.agent_i_gymunity_mapping["all_relative"][agent_i],
                self.agent_i_gymunity_mapping["team_relative"][agent_i]
            )
            self.agent_i_gymunity_mapping["all_relative"][agent_i] += self.agent_i_gymunity_mapping["team_relative"][agent_i]

        self.action_space = dcopy(self.env.action_space)

        self.observation_spaces = {}
        agent_i_gymunity = 0
        for multi_agent_ob in self.multi_agent_obs:
            for sensor in self.sensors:

                # BUG:
                observation_space = dcopy(
                    self.env.observation_space
                )

                observation_space.shape = replace_in_tuple(
                    tup=observation_space.shape,
                    index=self.dimension_to_cat_multi_agent_obs_for_sensor[
                        sensor
                    ],
                    value=observation_space.shape[
                        self.dimension_to_cat_multi_agent_obs_for_sensor[sensor]
                    ] *
                    len(
                        self.agent_i_gymunity_mapping[multi_agent_ob][agent_i_gymunity]
                    ),
                )
                self.observation_spaces["{}-{}".format(
                    multi_agent_ob,
                    sensor,
                )] = observation_space

        self.observation_space = try_reduce_dict(self.observation_spaces)
        if isinstance(self.observation_space, dict):
            self.observation_space = gym.spaces.Dict(self.observation_space)

    def sync_agent_i_gymunity2rllib(self):
        """sync agent_i_gymunity2rllib with agent_i_rllib2gymunity

        """
        for agent_i_gymunity in range(self.number_agents):
            self.agent_i_gymunity2rllib[agent_i_gymunity] = np.where(
                self.agent_i_rllib2gymunity == agent_i_gymunity
            )[0][0]

    def shuffle_agent_mapping(self):
        np.random.shuffle(self.agent_i_rllib2gymunity)
        self.sync_agent_i_gymunity2rllib()

    def run_an_episode(self, actions=None):
        """Run an episode with actions at each step.
        If actions is not provided, use self.action_space.sample() instead.
        """
        self.reset()

        if actions is None:
            actions = {}
            for agent_i in range(self.number_agents):
                actions[
                    agent_i2id(agent_i)
                ] = self.action_space.sample()

        while True:
            _, _, dones, _ = self.step(actions)
            if dones["__all__"]:
                break

    def reset(self):

        if self.is_shuffle_agents:
            self.shuffle_agent_mapping()

        obs_gymunity = self.env.reset()

        obs_rllib = self.obs_gymunity2rllib(obs_gymunity)

        return obs_rllib

    def step(self, actions_rllib):

        # actions_rllib to actions_gymunity
        actions_gymunity = self.actions_rllib2gymunity(actions_rllib)

        # step forward (gym_unity)
        obs_gymunity, rewards_gymunity, dones_gymunity, infos_gymunity = self.env.step(
            actions_gymunity
        )

        # returns_gymunity to returns_rllib
        obs_rllib, rewards_rllib, dones_rllib, infos_rllib = self.returns_gymunity2rllib(
            obs_gymunity, rewards_gymunity, dones_gymunity, infos_gymunity
        )

        # auto reset (rllib)
        if dones_rllib["__all__"] and ArenaRllibEnv.IS_AUTO_RESET:
            obs_rllib = self.reset()

        return obs_rllib, rewards_rllib, dones_rllib, infos_rllib

    def obs_gymunity2rllib(self, obs_gymunity):
        """Process obs_gymunity to obs_rllib.
        obs_gymunity: [sensor, multiple agents, (multiple visual observations,), ...]
        """

        obs = {}
        for agent_i_gymunity in range(self.number_agents):
            agent_i_rllib = self.agent_i_gymunity2rllib[agent_i_gymunity]
            agent_id_rllib = agent_i2id(agent_i_rllib)
            obs[agent_id_rllib] = {}
            for multi_agent_ob in self.multi_agent_obs:
                for sensor in self.sensors:
                    # [sensor, multiple agents, (multiple visual observations,), ...]
                    # get sensor
                    obs_ = obs_gymunity[
                        sensor.split("_")[0]
                    ]
                    # [multiple agents, (multiple visual observations,), ...]
                    # for visual observations, take one out
                    if sensor.split("_")[0] == "visual":
                        obs_ = obs_[
                            :, ArenaRllibEnv.VISUAL_SENSOR_INDEX[sensor]
                        ]
                    # [multiple agents, ...]
                    # take other agents obs according to agent_i_gymunity_mapping
                    obs_ = np.take(
                        obs_,
                        indices=self.agent_i_gymunity_mapping[multi_agent_ob][agent_i_gymunity],
                        axis=0,
                    )
                    # [selecyed multiple agents, ...]
                    obs_ = np.concatenate(
                        obs_,
                        axis=self.dimension_to_cat_multi_agent_obs_for_sensor[sensor],
                    )
                    # [..., selecyed multiple agents, ...]
                    obs[agent_id_rllib]["{}-{}".format(
                        multi_agent_ob,
                        sensor,
                    )] = obs_
            obs[agent_id_rllib] = try_reduce_dict(obs[agent_id_rllib])

        return obs

    def returns_gymunity2rllib(self, obs_gymunity, rewards_gymunity, dones_gymunity, infos_gymunity):
        """Process returns_gymunity to returns_rllib.
        """

        obs_rllib = self.obs_gymunity2rllib(obs_gymunity)

        rewards_rllib = {}
        dones_rllib = {}
        infos_rllib = {}
        for agent_i_gymunity in range(self.number_agents):
            agent_i_rllib = self.agent_i_gymunity2rllib[agent_i_gymunity]
            agent_id_rllib = agent_i2id(agent_i_rllib)
            rewards_rllib[agent_id_rllib] = rewards_gymunity[agent_i_gymunity]
            dones_rllib[agent_id_rllib] = dones_gymunity[agent_i_gymunity]
            infos_rllib[agent_id_rllib] = infos_gymunity

        # done when all agents are done
        dones_rllib["__all__"] = np.all(dones_gymunity)

        return obs_rllib, rewards_rllib, dones_rllib, infos_rllib

    def actions_rllib2gymunity(self, actions_rllib):
        """Process actions_rllib to actions_gymunity.
        """
        actions_gymunity = [None] * self.number_agents
        for agent_i_rllib in range(self.number_agents):
            agent_id_rllib = agent_i2id(agent_i_rllib)
            agent_i_gymunity = self.agent_i_rllib2gymunity[agent_i_rllib]
            actions_gymunity[agent_i_gymunity] = actions_rllib[agent_id_rllib]
        return actions_gymunity

    def close(self):
        self.env.close()


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

        return multiple_visual_obs

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

    def _multi_step(self, info):
        """arena-spec: add support for returning both vector observations and visual observations
        """
        default_observation = {}
        self.visual_obs = self._preprocess_multi(info.visual_observations)
        default_observation["visual"] = self.visual_obs
        default_observation["vector"] = info.vector_observations
        return (
            default_observation,
            info.rewards,
            info.local_done,
            {"text_observation": info.text_observations, "brain_info": info},
        )

    def _single_step(self, info):
        raise NotImplementedError


def get_env_directory(env_name):
    """Get env path according to env_name
    """
    return os.path.join(
        os.path.dirname(__file__),
        "bin/{}-{}".format(
            remove_arena_env_prefix(
                env_name
            ),
            platform.system(),
        )
    ), {
        "Linux": ".x86_64",
        "Darwin": ".app",
    }[platform.system()]


def remove_arena_env_prefix(env):
    """Remove ARENA_ENV_PREFIX from a env (possibly a grid_search).
    """
    env = dcopy(env)
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
