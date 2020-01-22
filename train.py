#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import copy
import os
import pickle
import logging
import utils
import glob

import numpy as np

import ray
from ray.tests.cluster_utils import Cluster
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments
from ray.rllib.utils.debug import summarize

from arena import *


class Trainer(ray.rllib.agents.trainer.Trainer):
    """Override Trainer so that it allow new configs."""
    _allow_unknown_configs = True


ray.rllib.agents.trainer.Trainer = Trainer

logger = logging.getLogger(__name__)


def run(args, parser):

    # create exps from configs
    if args.config_file:
        # load configs from yaml
        with open(args.config_file) as f:
            exps = yaml.safe_load(f)

    else:
        exps = create_exps(
            args=args,
        )

    arena_exps = create_arena_exps(
        exps=exps,
        args=args,
        parser=parser,
    )

    # config ray cluster
    if args.ray_num_nodes:
        cluster = Cluster()
        for ray_node in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory,
            )
        ray.init(
            address=cluster.redis_address,
        )
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus,
        )

    if len(arena_exps.keys()) > 1:
        logger.warning(
            "There are multiple experiments scheduled, ray==0.7.4 will run them one by one, instead of cocurrently. "
            "However, recent ray can run them cocurrently. But the recent ray has failed our test (the rllib is broken)"
            "This is mainly due to there are grid search used in configs that is not supported by original rllib. "
        )

    if args.eval:

        # evaluate policies

        if len(arena_exps.keys()) < 1:
            raise ValueError

        elif len(arena_exps.keys()) >= 1:

            if len(arena_exps.keys()) > 1:

                arena_exp_key = human_select(
                    options=list(arena_exps.keys()),
                    key="arena_exp_key",
                )

            else:
                # if there is just one arena_exps
                arena_exp_key = list(arena_exps.keys())[0]

        logger.info("Evaluating arena_exp_key: {}".format(
            arena_exp_key,
        ))

        arena_exp = arena_exps[arena_exp_key]

        from ray.rllib.evaluation.rollout_worker import RolloutWorker

        worker = RolloutWorker(
            env_creator=lambda _: ArenaRllibEnv(
                env=arena_exp["env"],
                env_config=arena_exp["config"]["env_config"],
            ),
            policy=arena_exp["config"]["multiagent"]["policies"],
            policy_mapping_fn=arena_exp["config"]["multiagent"]["policy_mapping_fn"],
            batch_mode="complete_episodes",
            batch_steps=500,
            num_envs=1,
        )

        for policy_id, policy in worker.policy_map.items():

            logdir = dcopy(args.eval_logdir)

            if logdir is None:
                logdir = human_select(
                    options=get_possible_logdirs(),
                    prefix_msg="Setting policy {}.".format(
                        policy_id,
                    ),
                    key="logdir",
                )
            else:
                if not os.path.exists(logdir):
                    raise Exception("logdir={} does not exist. ".format(
                        logdir
                    ))

            possible_populations = get_possible_populations(
                logdir=logdir
            )

            population_i = human_select(
                options=get_possible_populations(
                    logdir=logdir
                ),
                prefix_msg="Setting policy {}.".format(
                    policy_id,
                ),
                key="population_i",
            )

            iteration_i = human_select(
                options=get_possible_iterations(
                    logdir=logdir,
                    population_i=population_i,
                ),
                prefix_msg="Setting policy {}.".format(
                    policy_id,
                ),
                key="iteration_i",
            )

            checkpoint_path = get_checkpoint_path(
                logdir=logdir,
                population_i=population_i,
                iteration_i=iteration_i,
            )

            policy.set_weights(
                pickle.load(
                    open(
                        checkpoint_path,
                        "rb"
                    )
                )
            )

        while True:

            sample_batch = worker.sample()
            summarization = summarize_sample_batch(sample_batch)
            input(summarize(summarization))

    else:

        run_experiments(
            arena_exps,
            scheduler=_make_scheduler(args),
            queue_trials=args.queue_trials,
            resume=args.resume,
        )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
