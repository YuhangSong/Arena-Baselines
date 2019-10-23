import os
import time
import numpy as np

from envs_layer import ArenaMultiAgentEnvs


def main():
    num_envs = 3
    envs = ArenaMultiAgentEnvs(
        env_name='Arena-Test-Discrete',
        num_envs=num_envs,
        train_mode=False,
    )
    # ppo2.learn(
    #     network="mlp",
    #     env=env,
    #     total_timesteps=100000,
    #     lr=1e-3,
    # )
    obs = envs.reset()
    k = 0
    while True:
        actions = []
        for i in range(num_envs):
            action = np.random.randint(
                envs.action_space.n, size=envs.number_agents)
            action = action.tolist()
            actions += [action]
        obs, reward, done, info = envs.step(actions)
        print(np.shape(obs))
        print(reward)
        print(np.shape(done))
        print(done)
        print(info)
        k += 1
        print(k)


if __name__ == '__main__':
    main()
