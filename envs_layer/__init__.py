import time


class ArenaMultiAgentEnvs(object):
    """docstring for ArenaMultiAgentEnvs."""

    def __init__(
        self,
        env_name,
        num_envs,
        train_mode=True,
    ):
        """
        ArenaMultiAgentEnvs initialization
        :param env_name: Name of the environment.
        :param num_envs: Worker number for environment.
        :param train_mode: Whether to run in training mode, speeding up the simulation, by default.
        """

        super(ArenaMultiAgentEnvs, self).__init__()
        self.platform = env_name.split('-')[0]
        self.env_name = env_name.split(self.platform + '-')[1]
        self.num_envs = num_envs
        self.train_mode = train_mode

        if self.platform in 'Arena':
            from .arena_envs import get_env_directory
            from .arena_envs import arena_make_unity_env as make_unity_env
            self.envs = make_unity_env(
                env_directory=get_env_directory(self.env_name),
                num_env=self.num_envs,
                visual=True,
                start_index=int(time.time()) % 65534,
                train_mode=self.train_mode,
            )
            self.action_space = self.envs.action_space
            self.observation_space = self.envs.observation_space
            self.number_agents = self.envs.number_agents

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (numpy.ndarray): dtype: depends, shape: (self.num_envs, self.number_agents, h, w, c)
        """
        return self.envs.reset()

    def step(self, actions):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (list): [[action (int)]*self.number_agents]*self.num_envs
            Example to generate actions:
                actions = []
                for i in range(num_envs):
                    action = np.random.randint(
                        envs.action_space.n, size=envs.number_agents)
                    action = action.tolist()
                    actions += [action]
        Returns:
            observation (numpy.ndarray): agent's observation of the current environment, dtype: depends, shape: (self.num_envs, self.number_agents, h, w, c)
            reward (numpy.ndarray) : amount of reward returned after previous action, dtype: float, shape: (self.num_envs, self.number_agents)
            done (numpy.ndarray): whether the episode has ended, dtype: bool, shape: (self.num_envs, self.number_agents)
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """
        return self.envs.step(actions)
