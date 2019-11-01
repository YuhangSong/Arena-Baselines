import os
import time
import numpy as np

from envs_layer import ArenaRllibEnv


def main():
    env = ArenaRllibEnv()

    # Observations are a dict mapping agent names to their obs. Not all agents
    # may be present in the dict in each time step.
    # print(env.reset())
    # {
    #     "Player_0": [[...]],
    #     "Player_1": [[...]],
    # }

    while True:
        # Actions should be provided for each agent that returned an observation.
        new_obs, rewards, dones, infos = env.step(
            actions={"Player_0": 0, "Player_1": 5})

        # Similarly, new_obs, rewards, dones, etc. also become dicts
        print(rewards)
        # {"Player_0": 3, "Player_1": -1}

        # Individual agents can early exit; env is done when "__all__" = True
        print(dones)
        # {"Player_0": True, "Player_1": False,, "__all__": True}


if __name__ == '__main__':
    main()
