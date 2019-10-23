import os
import time
import numpy as np

from envs_layer import ArenaMultiAgentEnvs


def main():
    num_envs = 3
    ma_envs = ArenaMultiAgentEnvs(
        env_name='Arena-Test-Discrete',
        num_envs=num_envs,
        train_mode=False,
    )

    # single-agent-like sync step

    for agent_i in range(ma_envs.number_agents):
        print('A{}: spaces {} {}'.format(
            agent_i, ma_envs.sa_envs[agent_i].action_space, ma_envs.sa_envs[agent_i].observation_space))

    for agent_i in range(ma_envs.number_agents):
        ma_envs.sa_envs[agent_i].reset()
        print('A{}: reset'.format(agent_i))
    k = 0
    for agent_i in range(ma_envs.number_agents):
        obs = ma_envs.sa_envs[agent_i].observe_after_reset()
        print('A{}: observe_after_reset {} {}'.format(
            agent_i, type(obs), np.shape(obs)))
    while True:
        for agent_i in range(ma_envs.number_agents):
            actions = np.random.randint(
                ma_envs.sa_envs[agent_i].action_space.n, size=(ma_envs.num_envs))
            print('A{}: act {}'.format(agent_i, actions))
            ma_envs.sa_envs[agent_i].step(actions)
        ma_envs.step_sync()
        print('All agents step sync'.format())
        for agent_i in range(ma_envs.number_agents):
            obs, reward, done, info = ma_envs.sa_envs[agent_i].observe_after_step(
            )
            print('A{}: observe_after_step {} {} {} {} {} {}'.format(
                agent_i,
                type(obs), np.shape(obs),
                type(reward), np.shape(reward),
                type(done), np.shape(done),
            ))
        k += 1
        print('step at {}'.format(k))

    # # multi-agent step
    # obs = ma_envs.reset()
    # k = 0
    # while True:
    #     actions = np.random.randint(ma_envs.action_space.n, size=(
    #         ma_envs.num_envs, ma_envs.number_agents))
    #     print('act {}'.format(actions))
    #     obs, reward, done, info = ma_envs.step(actions)
    #     print('step {} {} {} {} {} {}'.format(
    #         type(obs), np.shape(obs),
    #         type(reward), np.shape(reward),
    #         type(done), np.shape(done),
    #     ))
    #     k += 1
    #     print('step at {}'.format(k))


if __name__ == '__main__':
    main()
