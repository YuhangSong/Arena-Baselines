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

        elif len(arena_exps.keys()) > 1:

            # if there are multiple arena_exps, select one

            selection_dict = {}
            for i in range(len(arena_exps.keys())):
                selection_dict[i] = list(arena_exps.keys())[i]

            input_ = input(
                "WARNING: There are multiple arena_exps as follows: \n{} \nPlease select one of them by number (0-{}):".format(
                    dict_to_print_str(selection_dict),
                    len(arena_exps.keys()) - 1,
                )
            )

            selected_i = int(input_)

        else:

            # if there is just one arena_exps
            selected_i = 0

        selected_key = list(arena_exps.keys())[selected_i]

        logger.info("Evaluating arena_exp: {}".format(
            selected_key,
        ))

        config = arena_exps[selected_key]

        from ray.rllib.evaluation.rollout_worker import RolloutWorker

        worker = RolloutWorker(
            env_creator=lambda _: ArenaRllibEnv(
                env=config["env"],
                env_config=update_config_value_by_key_value(
                    config_to_update=config["config"]["env_config"],
                    config_key="train_mode",
                    config_value=False,
                ),
            ),
            policy=config["config"]["multiagent"]["policies"],
            policy_mapping_fn=config["config"]["multiagent"]["policy_mapping_fn"],
        )

        while True:
            if args.eval_logdir is None:
                possible_logdirs = get_possible_logdirs()
                args.eval_logdir = input(
                    "args.eval_logdir is required, you can type it in now:")
            else:
                if os.path.exists(args.eval_logdir):
                    break
                else:
                    logger.warning("args.eval_logdir={} does not exist. You will be promoted to choose one that exists. ".format(
                        args.eval_logdir
                    ))
                    args.eval_logdir = None

        possible_populations = get_possible_populations(
            logdir=args.eval_logdir)

        while True:
            population_i = int(input("possible_populations are {}, choose one of them:".format(
                possible_populations
            )))
            if population_i in possible_populations:
                break
            else:
                logger.warning(
                    "population_i should be in possible_populations")

        possible_iterations = get_possible_iterations()

        while True:
            iteration_i = int(input("possible_iterations are {}, choose one of them:".format(
                possible_populations
            )))
            if iteration_i in possible_iterations:
                break
            else:
                logger.warning("iteration_i should be in possible_iterations")

        checkpoint_path = get_checkpoint_path(
            logdir=args.eval_logdir,
            population_i=population_i,
            iteration_i=iteration_i,
        )

        logger.info("loading from checkpoint_path: {}".format(
            checkpoint_path
        ))

        input(worker.policy_map)
        input(worker.sample())

    else:

        run_experiments(
            arena_exps,
            scheduler=_make_scheduler(args),
            queue_trials=args.queue_trials,
            resume=args.resume,
        )


if __name__ == "__main__":
    get_possible_logdirs()
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
