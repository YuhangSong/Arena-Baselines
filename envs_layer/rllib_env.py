import os
import platform
import numpy as np

from gym_unity.envs import UnityEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def get_env_directory(env_name):
    '''
        get_env_directory:
            get env path according to env_name
    '''
    return os.path.join(
        os.path.dirname(__file__),
        'bin/{}-{}'.format(
            env_name,
            platform.system(),
        )
    )


class ArenaUnityEnv(UnityEnv):

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

        rank = 0
        while True:
            try:
                self.env = ArenaUnityEnv(
                    get_env_directory(self.env_config['name']),
                    rank,
                    use_visual=True,
                    uint8_visual=False,
                    multiagent=True,
                )
                break
            except Exception as e:
                rank += 1

        self.env.set_train_mode(train_mode=self.env_config['train_mode'])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.player_name_prefix = 'P'

    def reset(self):

        obs_ = self.env.reset()

        # xxx_(gym_unity) to xxx(rllib)
        obs = {}
        for i in range(len(obs)):
            obs['{}{}'.format(self.player_name_prefix, i)] = obs_[i]

        return obs

    def step(self, actions):

        # xxx(rllib) to xxx_(gym_unity)
        actions_ = []
        for i in range(len(actions.keys())):
            actions_ += [actions['{}{}'.format(self.player_name_prefix, i)]]

        print(len(actions_))
        print(type(actions_))
        obs_, rewards_, dones_, infos_ = self.env.step(actions_)

        # xxx_(gym_unity) to xxx(rllib)
        obs = {}
        rewards = {}
        dones = {}
        infos = infos_
        for i in range(len(obs_)):
            obs['{}{}'.format(self.player_name_prefix, i)] = obs_[i]
            rewards['{}{}'.format(self.player_name_prefix, i)] = rewards_[i]
            dones['{}{}'.format(self.player_name_prefix, i)] = dones_[i]

        # done when all agents are done
        dones['__all__'] = np.all(dones_)

        return obs, rewards, dones, infos

    def close(self):
        self.env.close()
