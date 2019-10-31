import os
import time
import numpy as np

from envs_layer import ArenaRllibEnv


def main():
    env = ArenaRllibEnv()

    # Observations are a dict mapping agent names to their obs. Not all agents
    # may be present in the dict in each time step.
    print(env.reset())
    # {
    #     "car_1": [[...]],
    #     "car_2": [[...]],
    #     "traffic_light_1": [[...]],
    # }

    while True:
        # Actions should be provided for each agent that returned an observation.
        new_obs, rewards, dones, infos = env.step(actions={"P0": 0, "P1": 5})

        # Similarly, new_obs, rewards, dones, etc. also become dicts
        print(rewards)
        # {"car_1": 3, "car_2": -1, "traffic_light_1": 0}

        # Individual agents can early exit; env is done when "__all__" = True
        print(dones)
        # {"car_2": True, "__all__": False}


if __name__ == '__main__':
    main()
