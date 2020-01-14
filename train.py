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

import arena


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
        exps = arena.create_exps(
            args=args,
        )

    arena_exps = arena.create_arena_exps(
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

    run_experiments(
        arena_exps,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
    )


if __name__ == "__main__":
    parser = arena.create_parser()
    args = parser.parse_args()
    run(args, parser)
