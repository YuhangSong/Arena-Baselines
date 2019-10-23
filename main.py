import os
import time
import numpy as np

from envs_layer import ArenaMultiAgentEnvs


def main():
    num_envs = 3
    envs = ArenaMultiAgentEnvs(
        env_name='Arena-Test-Discrete',
        num_envs=num_envs,
    )
    # ppo2.learn(
    #     network="mlp",
    #     env=env,
    #     total_timesteps=100000,
    #     lr=1e-3,
    # )
    print(envs.reset())
    while True:
        actions = []
        for i in range(num_envs):
            action = np.random.randint(
                envs.action_space.n, size=envs.number_agents)
            action = action.tolist()
            actions += [action]
        envs.step(actions)
        print('ss')


if __name__ == '__main__':
    main()
