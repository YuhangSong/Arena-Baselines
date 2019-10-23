import time
import numpy as np


class ArenaAgent(object):
    """docstring for ArenaAgent."""

    def __init__(
        self,
        action_space,
        observation_space,
        num_envs,
        id,
    ):
        """
        ArenaAgent initialization
        :param xx: xx.
        """

        super(ArenaAgent, self).__init__()

        self.action_space = action_space
        self.observation_space = observation_space
        self.num_envs = num_envs
        self.id = id

        print('ID{}: [agent] spaces {} {}'.format(
            self.id, self.action_space, self.observation_space))

    def observe_after_reset(self, obs):
        print('ID{}: [agent] observe_after_reset {} {}'.format(
            self.id, type(obs), np.shape(obs)))

    def act(self):
        actions = np.random.randint(self.action_space.n, size=(self.num_envs))
        print('ID{}: [agent] act {}'.format(self.id, actions))
        return actions

    def observe_after_step(self, obs, reward, done, info):
        print('ID{}: [agent] observe_after_step {} {} {} {} {} {}'.format(
            self.id,
            type(obs), np.shape(obs),
            type(reward), np.shape(reward),
            type(done), np.shape(done),
        ))


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
