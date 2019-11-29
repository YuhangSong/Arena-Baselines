from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Simple example of using ArenaRllibEnv, which is a interface that
convert a arena environment to a MultiAgentEnv
(see: https://ray.readthedocs.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
interface by rllib.
"""

import numpy as np

from envs_layer import ArenaRllibEnv


def main():

    env_config = {
        "env_id": "Tennis-Sparse-2T1P-Discrete",
        "is_shuffle_agents": True,
        "train_mode": False,
    }
    env = ArenaRllibEnv(env_config)
    new_obs = nev.reset()

    # Observations are a dict mapping agent names to their obs. Not all agents
    # may be present in the dict in each time step.
    # print(env.reset())
    # {
    #     "agent_0": [[...]],
    #     "agent_1": [[...]],
    # }

    episode_video = None

    while True:

        new_obs_shapes = {}
        for key in new_obs.keys():
            new_obs_shapes[key] = np.shape(new_obs[key])

        input("new_obs_shapes: {}; dtype: {}; min: {}; max: {}".format(
            new_obs_shapes,
            new_obs.dtype,
            np.min(new_obs),
            np.max(new_obs),
        ))
        # new_obs_shapes: {'agent_0': (84, 84, 1), 'agent_1': (84, 84, 1)}

        temp = np.expand_dims(new_obs["agent_0"][:, :, 0], 0)
        if episode_video is None:
            episode_video = temp
        else:
            episode_video = np.concatenate((episode_video, temp))

        # Actions should be provided for each agent that returned an observation.
        new_obs, rewards, dones, infos = env.step(
            actions={"agent_0": 0, "agent_1": 5})

        print("rewards: {}".format(rewards))
        # rewards: {"agent_0": 3, "agent_1": -1}

        # Individual agents can early exit; env is done when "__all__" = True
        print("dones: {}".format(dones))
        # dones: {"agent_0": True, "agent_1": False,, "__all__": True}

        print("infos: {}".format(infos))

        if dones["__all__"]:
            # initialize video writer
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            fps = 15
            video_filename = '../episode_video.avi'
            video_size = (episode_video[key].size()[4],
                          episode_video[key].size()[3])
            video_writer = cv2.VideoWriter(
                video_filename, fourcc, fps, video_size)

            for frame_i in range(episode_video[key].size()[1]):
                gray = episode_video[key].squeeze(
                    0)[frame_i].squeeze(0).cpu().numpy().astype(np.uint8)
                gray_3c = cv2.merge([gray, gray, gray])
                # np.shape([3, H, W]), 0-255, np.uint8
                video_writer.write(
                    gray_3c
                )

            video_writer.release()
            episode_video = None
            input('episode end, keep going?')
            env.reset()


if __name__ == '__main__':
    main()
