import time
import numpy as np


class ArenaSalAgent(object):
    """docstring for ArenaSalAgent."""

    def __init__(
        self,
        action_space,
        observation_space,
        num_envs,
        id,
    ):
        """
        ArenaSalAgent initialization
        :param xx: xx.
        """

        super(ArenaSalAgent, self).__init__()

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
