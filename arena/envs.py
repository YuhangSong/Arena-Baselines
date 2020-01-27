from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym_unity.envs import UnityEnv
from gym import error, spaces

from .utils import *
from .constants import *

logger = logging.getLogger(__name__)

IS_AUTO_RESET = True
VALID_SENSORS = ["visual_FP", "visual_TP", "vector"]
SENSOR2CAMERA = {
    "visual_FP": 0,
    "visual_TP": 1,
}
CAMERA2SENSOR = dict([(value, key) for key, value in SENSOR2CAMERA.items()])


def _validate_sensors(sensors):
    assert isinstance(sensors, list)
    for sensor in sensors:
        if sensor not in VALID_SENSORS:
            raise Exception("sensor {} is invalid".format(sensor))


class ArenaRllibEnv(MultiAgentEnv):
    """Convert ArenaUnityEnv(gym_unity) to MultiAgentEnv (rllib)

        The action_space and observation_space are shared across agents.

        The config sensors and multi_agent_obs are two lists.
        The observation_space is determined by the joint value of them.
        If len(sensors) * len(multi_agent_obs)>1, the observation_space would be a gym.spaces.Dict,
        where the keys are multi_agent_ob-sensor, with multi_agent_ob and sensor being the element
        in multi_agent_obs and sensors.
    """

    """Following configurations need to be compatible with Arena-BuildingToolkit.
    """

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
        _validate_sensors(self.sensors)

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
                observation_space = dcopy(
                    self.env.observation_space[sensor]
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
        if dones_rllib["__all__"] and IS_AUTO_RESET:
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
                            :, SENSOR2CAMERA[sensor]
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

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def spec(self):
        return self.env.spec

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self


class ArenaUnityEnv(UnityEnv):
    """An override of UnityEnv from gym_unity.envs, to fix some of their bugs and add some supports.
    Search "arena-spec" for these places.
    """

    def __init__(self, *args, **kwargs):
        """arena-spec: add support for multiple sensors, observation_space is modified to be a dict
        """
        super(ArenaUnityEnv, self).__init__(*args, **kwargs)

        brain = self._env.brains[self.brain_name]

        self._observation_space = {}

        for camera_i in range(len(brain.camera_resolutions)):
            if brain.camera_resolutions[camera_i]["blackAndWhite"]:
                depth = 1
            else:
                depth = 3
            self._observation_space[CAMERA2SENSOR[camera_i]] = spaces.Box(
                0,
                1,
                dtype=np.float32,
                shape=(
                    brain.camera_resolutions[camera_i]["height"],
                    brain.camera_resolutions[camera_i]["width"],
                    depth,
                ),
            )

        high = np.array([np.inf] * brain.vector_observation_space_size)
        self._observation_space['vector'] = spaces.Box(
            -high, high, dtype=np.float32)

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

    def render(self, mode="rgb_array"):
        """arena-spec: add support for rendering visual_obs of multiple agents and multiple cameras into one grided rendered frame
        """

        logger.info("rendering")

        if mode in ['rgb_array']:

            if len(np.shape(self.visual_obs)) == 5:
                # (multiple agents, multiple visual obs, 84, 84, 1)

                frame = self.visual_obs

                # convert to uint8
                if frame.dtype != np.uint8:
                    frame = (255.0 * frame).astype(np.uint8)

                # reshape to (multiple_img, img_shape)
                frame = np.reshape(
                    frame,
                    newshape=(-1,) + np.shape(self.visual_obs)[2:],
                )

                # convert to rgb
                if np.shape(frame)[-1] == 1:
                    frame = np.concatenate(
                        [frame] * 3,
                        axis=3,
                    )

                # grid them to make it a gallery single img
                frame = gallery(
                    frame,
                    ncols=self.number_agents,
                )

                return frame

            else:
                raise NotImplementedError

        else:
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
