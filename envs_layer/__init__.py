import time
import numpy as np


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

        # single agent envs
        self.sa_envs = [ArenaSingleAgentHolderEnvs(
            action_space=self.action_space,
            observation_space=self.observation_space,
            master_multi_agent_envs=self,
            id=id,
        ) for id in range(self.number_agents)]

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (numpy.ndarray): dtype: depends, shape: (self.num_envs, self.number_agents, h, w, c)
        """
        self.observation = self.envs.reset()
        return self.observation

    def step(self, actions):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            actions (numpy.ndarray) : dtype: int, shape: (self.num_envs, self.number_agents)
            Example to generate actions:
                actions = np.random.randint(self.action_space.n, size=(self.num_envs, self.number_agents))
        Returns:
            observation (numpy.ndarray): agent's observation of the current environment, dtype: depends, shape: (self.num_envs, self.number_agents, h, w, c)
            reward (numpy.ndarray) : amount of reward returned after previous action, dtype: float, shape: (self.num_envs, self.number_agents)
            done (numpy.ndarray): whether the episode has ended, dtype: bool, shape: (self.num_envs, self.number_agents)
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """
        self.observation, self.reward, self.done, self.infos = self.envs.step(
            actions.tolist())
        return self.observation, self.reward, self.done, self.infos

    def step_sync(self):
        actions = [np.expand_dims(self.sa_envs[id].actions, axis=1)
                   for id in range(self.number_agents)]
        actions = np.concatenate(actions, axis=1)
        self.step(actions)


class ArenaSingleAgentHolderEnvs(object):
    """docstring for ArenaSingleAgentHolderEnvs."""

    def __init__(self, action_space, observation_space, master_multi_agent_envs, id):
        super(ArenaSingleAgentHolderEnvs, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.master_multi_agent_envs = master_multi_agent_envs
        self.id = id

    def reset(self):
        """Resets the state of the environment. Reset will cause master_multi_agent_envs to reset.
        """
        return self.master_multi_agent_envs.reset()

    def observe_after_reset(self):
        """Observe after calling reset.
        Returns: observation (numpy.ndarray): dtype: depends, shape: (self.num_envs, h, w, c)
        """
        return self.master_multi_agent_envs.observation[:, self.id]

    def step(self, actions):
        """Run one timestep of the environment's dynamics.
        Accepts an action.
        Args:
            actions (numpy.ndarray) : dtype: int, shape: (self.num_envs)
            Example to generate actions:
                actions = np.random.randint(self.action_space.n, size=(self.num_envs))
        """
        self.actions = actions

    def observe_after_step(self):
        """Observe after calling master_multi_agent_envs.step_sync.
        Returns:
            observation (numpy.ndarray): agent's observation of the current environment, dtype: depends, shape: (self.num_envs, h, w, c)
            reward (numpy.ndarray) : amount of reward returned after previous action, dtype: float, shape: (self.num_envs)
            done (numpy.ndarray): whether the episode has ended, dtype: bool, shape: (self.num_envs)
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """
        return self.master_multi_agent_envs.observation[:, self.id], self.master_multi_agent_envs.reward[:, self.id], self.master_multi_agent_envs.done[:, self.id], self.master_multi_agent_envs.infos
