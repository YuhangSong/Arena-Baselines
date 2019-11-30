from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Simple example of using ArenaRllibEnv, which is a interface that
convert a arena environment to a MultiAgentEnv
(see: https://ray.readthedocs.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
interface by rllib.
"""

import yaml
import cv2

import numpy as np

from envs_layer import ArenaRllibEnv

from train import create_parser


def run(args, parser):

    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)

    env_config = experiments["test-arena"]["config"]["env_config"]

    env = ArenaRllibEnv(env_config)

    new_obs = env.reset()

    if "visual" in env_config["obs_type"]:
        episode_video = None

    while True:

        new_obs_infos = {}
        for key in new_obs.keys():
            new_obs_infos[key] = "shape: {}; dtype: {}; min: {}; max: {}".format(
                np.shape(new_obs[key]),
                new_obs[key].dtype,
                np.min(new_obs[key]),
                np.max(new_obs[key]),
            )

        # Observations are a dict mapping agent names to their obs. Not all agents
        # may be present in the dict in each time step.
        print("new_obs_infos: {}".format(new_obs_infos))
        # ew_obs_infos: {'agent_0': 'shape: (84, 84, 1); dtype: float64; min: 0.0; max: 0.9098039215686274', 'agent_1': 'shape: (84, 84, 1); dtype: float64; min: 0.0; max: 0.9098039215686274'}

        # record visual obs as video
        if "visual" in env_config["obs_type"]:
            temp = (np.expand_dims(
                new_obs["agent_0"][:, :, 0],
                0
            ) * 255.0
            ).astype(np.uint8)
            if episode_video is None:
                episode_video = temp
            else:
                episode_video = np.concatenate((episode_video, temp))

        # Actions should be provided for each agent that returned an observation.
        new_obs, rewards, dones, infos = env.step(
            actions={"agent_0": 0, "agent_1": 7})

        print("rewards: {}".format(rewards))
        # rewards: {"agent_0": 3, "agent_1": -1}

        # Individual agents can early exit; env is done when "__all__" = True
        print("dones: {}".format(dones))
        # dones: {"agent_0": True, "agent_1": False,, "__all__": True}

        print("infos: {}".format(infos))

        if dones["__all__"]:

            # record visual obs as video
            if "visual" in env_config["obs_type"]:
                # initialize video writer
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                fps = 15
                video_filename = '../episode_video.avi'
                video_size = (np.shape(episode_video)[2],
                              np.shape(episode_video)[1])
                video_writer = cv2.VideoWriter(
                    video_filename, fourcc, fps, video_size)

                for frame_i in range(np.shape(episode_video)[0]):
                    gray = episode_video[frame_i]
                    gray_3c = cv2.merge([gray, gray, gray])
                    # np.shape([3, H, W]), 0-255, np.uint8
                    video_writer.write(
                        gray_3c
                    )

                video_writer.release()
                episode_video = None

            input('episode end, keep going?')
            env.reset()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
