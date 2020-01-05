import os
import platform
import random
import time

import numpy as np

from gym_unity.envs import UnityEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

ARENA_ENV_PREFIX = 'Arena-'
AGENT_ID_PREFIX = "agent"
POLICY_ID_PREFIX = "policy"


class ArenaRllibEnv(MultiAgentEnv):
    """Convert ArenaUnityEnv(gym_unity) to MultiAgentEnv (rllib)
    """

    def __init__(self, env, env_config):

        self.env = env
        if self.env is None:
            raise Exception("env in has to be specified")

        self.obs_type = env_config.get("obs_type", "visual_FP")

        if "-" in self.obs_type:
            input('# TODO: multiple obs support')

        if self.obs_type in ["visual_TP"]:
            input('# TODO: visual_TP obs support')

        game_file_path = get_env_directory(self.env)

        if self.obs_type in ["vector"]:

            if os.path.exists(game_file_path + '-Server'):
                game_file_path = game_file_path + '-Server'
                input(
                    "# WARNING: Using server build"
                )

            else:
                input(
                    "# WARNING: only vector observation is used, you can have a server build which runs faster"
                )

        while True:
            try:
                # TODO: Individual game instance cannot get rank from rllib, so just try ranks
                rank = random.randint(0, 65534)
                self.env = ArenaUnityEnv(
                    game_file_path,
                    rank,
                    use_visual=False if self.obs_type in ["vector"] else True,
                    uint8_visual=False,
                    multiagent=True,
                )
                break
            except Exception as e:
                pass

        self.env.set_train_mode(train_mode=env_config.get("train_mode", True))

        self.is_shuffle_agents = env_config.get("is_shuffle_agents", False)
        if self.is_shuffle_agents:
            self.shift = 0

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.number_agents = self.env.number_agents

    def reset(self):

        if self.is_shuffle_agents:
            self.shift = np.random.randint(0, self.number_agents)

        obs_ = self.env.reset()

        if self.is_shuffle_agents:
            obs_ = self.roll_back(obs_)

        # xxx_ (gym_unity) to xxx (rllib)
        obs = {}
        for agent_i in range(self.number_agents):
            obs[self.get_agent_id(agent_i)] = obs_[agent_i]

        return obs

    def step(self, actions):

        # xxx (rllib) to xxx_ (gym_unity)
        actions_ = []
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            actions_ += [actions[agent_id]]

        if self.is_shuffle_agents:
            actions_ = self.roll(actions_).tolist()

        # step forward (gym_unity)
        obs_, rewards_, dones_, infos_ = self.env.step(actions_)

        if self.is_shuffle_agents:
            obs_ = self.roll_back(obs_)
            rewards_ = self.roll_back(rewards_)
            dones_ = self.roll_back(dones_)
            infos_["shift"] = self.shift

        # xxx_ (gym_unity) to xxx (rllib)
        obs = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent_i in range(self.number_agents):
            agent_id = self.get_agent_id(agent_i)
            obs[agent_id] = obs_[agent_i]
            rewards[agent_id] = rewards_[agent_i]
            dones[agent_id] = dones_[agent_i]
            infos[agent_id] = infos_

        # done when all agents are done
        dones["__all__"] = np.all(dones_)

        return obs, rewards, dones, infos

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


def get_env_directory(env_name):
    """get_env_directory:
        get env path according to env_name
    """
    return os.path.join(
        os.path.dirname(__file__),
        "bin/{}-{}".format(
            env_name,
            platform.system(),
        )
    )
