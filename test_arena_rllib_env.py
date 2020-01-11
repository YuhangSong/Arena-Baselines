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
import logging
import arena
import numpy as np

np.set_printoptions(edgeitems=1)

logger = logging.getLogger(__name__)


def run(args, parser):

    with open(args.config_file) as f:
        experiments = yaml.safe_load(f)

    env = arena.get_one_from_grid_search(
        arena.remove_arena_env_prefix(
            experiments["Arena-Benchmark"]["env"]
        )
    )
    env_config = experiments["Arena-Benchmark"]["config"]["env_config"]

    env_config["obs_type"] = arena.get_one_from_grid_search(
        env_config["obs_type"]
    )

    env_config["train_mode"] = False

    logger.info(env)
    # Tennis-Sparse-2T1P-Discrete
    logger.info(env_config)
    # {'is_shuffle_agents': True, 'train_mode': True, 'obs_type': 'visual_FP'}

    env = arena.ArenaRllibEnv(
        env=env,
        env_config=env_config,
    )

    logger.info(env.observation_space)
    logger.info(env.action_space)

    obs_rllib = env.reset()

    logger.info("obs_rllib: {}".format(obs_rllib))

    episode_video = {}

    while True:

        # Actions should be provided for each agent that returned an observation.
        obs_rllib, rewards_rllib, dones_rllib, infos_rllib = env.step(
            # actions={"agent_0": 0, "agent_1": 7}
            actions_rllib={
                "agent_0": 0,
                "agent_1": 5,
                "agent_2": 6,
                "agent_3": 3,
            }
        )

        logger.info("obs_rllib: {}".format(obs_rllib))
        logger.info("rewards_rllib: {}".format(rewards_rllib))
        logger.info("dones_rllib: {}".format(dones_rllib))
        logger.info("infos_rllib: {}".format(infos_rllib))

        if dones_rllib["__all__"]:

            for episode_video_key in episode_video.keys():

                # initialize video writer
                fourcc = cv2.VideoWriter_fourcc(
                    'M', 'J', 'P', 'G'
                )
                fps = 15
                video_filename = "../{}.avi".format(
                    episode_video_key,
                )
                video_size = (
                    np.shape(episode_video[episode_video_key])[2],
                    np.shape(episode_video[episode_video_key])[1]
                )
                video_writer = cv2.VideoWriter(
                    video_filename, fourcc, fps, video_size
                )

                for frame_i in range(np.shape(episode_video[episode_video_key])[0]):
                    video_writer.write(
                        episode_video[episode_video_key][frame_i]
                    )

                video_writer.release()

            episode_video = {}

            input('episode end, keep going?')

        else:

            for agent_id in obs_rllib.keys():

                obs_each_agent = obs_rllib[agent_id]

                if isinstance(obs_each_agent, dict):

                    obs_keys = obs_each_agent.keys()

                else:

                    obs_keys = ["default_own_obs"]

                for obs_key in obs_keys:

                    if isinstance(obs_each_agent, dict):
                        obs_each_key = obs_each_agent[obs_key]
                    else:
                        obs_each_key = obs_each_agent

                    obs_each_channel = {}

                    if len(np.shape(obs_each_key)) == 1:

                        # vector observation

                        obs_each_channel["default_channel"] = arena.get_img_from_fig(
                            arena.plot_feature(
                                obs_each_key
                            )
                        )

                    elif len(np.shape(obs_each_key)) == 3:

                        # visual observation

                        for channel_i in range(np.shape(obs_each_key)[2]):

                            gray = obs_each_key[
                                :, :, channel_i
                            ]

                            rgb = cv2.merge([gray, gray, gray])

                            rgb = (rgb * 255.0).astype(np.uint8)

                            obs_each_channel["{}_channel".format(
                                channel_i
                            )] = rgb

                    else:

                        raise NotImplementedError

                    for channel_key in obs_each_channel.keys():

                        temp = np.expand_dims(
                            obs_each_channel[channel_key],
                            0
                        )

                        episode_video_key = "agent_{}-obs_{}-channel-{}".format(
                            agent_id,
                            obs_key,
                            channel_key,
                        )

                        if episode_video_key not in episode_video.keys():
                            episode_video[episode_video_key] = temp
                        else:
                            episode_video[episode_video_key] = np.concatenate(
                                (episode_video[episode_video_key], temp)
                            )


if __name__ == "__main__":
    parser = arena.create_parser()
    args = parser.parse_args()
    run(args, parser)
