import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class ArenaRllibEnv(MultiAgentEnv):
    """Two-player environment for rock paper scissors.

    The observation is simply the last opponent action."""

    def __init__(
        self,
        env_config={
            'name': 'Test-Discrete',
            'train_mode': False,
        }
    ):

        self.env_config = env_config

        from .arena_envs import UnityEnv, get_env_directory
        self.env = UnityEnv(get_env_directory(self.env_config['name']), 0, use_visual=True,
                            uint8_visual=True, multiagent=True)
        self.env.set_train_mode(train_mode=self.env_config['train_mode'])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.player_name_prefix = 'P'

    def reset(self):
        return_ = {}
        obs = self.env.reset()
        for i in range(len(obs)):
            return_['{}{}'.format(self.player_name_prefix, i)] = obs[i]
        return return_

    def step(self, actions):

        actions_ = []
        for i in range(len(actions.keys())):
            actions_ += [actions['{}{}'.format(self.player_name_prefix, i)]]

        obs_, rewards_, dones_, infos_ = self.env.step(actions_)

        obs = {}
        rewards = {}
        dones = {}
        for i in range(len(obs_)):
            obs['{}{}'.format(self.player_name_prefix, i)] = obs_[i]
            rewards['{}{}'.format(self.player_name_prefix, i)] = rewards_[i]
            dones['{}{}'.format(self.player_name_prefix, i)] = dones_[i]

        dones['__all__'] = np.all(dones_)

        return obs, rewards, dones, infos_
