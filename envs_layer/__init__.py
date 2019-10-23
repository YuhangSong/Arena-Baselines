import time


class ArenaMultiAgentEnvs(object):
    """docstring for ArenaMultiAgentEnvs."""

    def __init__(self, env_name, num_envs):
        super(ArenaMultiAgentEnvs, self).__init__()
        self.platform = env_name.split('-')[0]
        self.env_name = env_name.split(self.platform + '-')[1]
        self.num_envs = num_envs

        if self.platform in 'Arena':
            from .arena_envs.arena_driver import make_envs
            self.envs = make_envs(
                env_name=self.env_name,
                num_envs=self.num_envs,
                visual=True,
                start_index=int(time.time()) % 65534,
            )
            self.action_space = self.envs.action_space
            self.observation_space = self.envs.observation_space
            self.number_agents = self.envs.number_agents

    def reset(self):
        return self.envs.reset()
