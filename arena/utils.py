import os
import re
import cv2
import io
import logging

import platform
import random
import gym
import json

from PyInquirer import style_from_dict, Token, prompt, Separator
from examples import custom_style_2

import copy
from copy import deepcopy as dcopy

import ray
from ray.rllib.utils.debug import summarize
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')


logger = logging.getLogger(__name__)


def summarize_sample_batch(sample_batch):

    summarization = {}

    if isinstance(sample_batch, MultiAgentBatch):

        for policy_id, sample_batch_per_policy in sample_batch.policy_batches.items():

            summarization[policy_id] = summarize_sample_batch_per_policy(
                sample_batch_per_policy)

    else:

        raise NotImplementedError

    return summarization


def summarize_sample_batch_per_policy(sample_batch_per_policy):

    summarization_per_policy = {}

    summarization_keys = [
        "episode_rewards",
        "episode_lengths",
    ]

    for sample_batch_per_policy_per_episode in sample_batch_per_policy.split_by_episode():
        sample_batch_per_policy_per_episode = sample_batch_per_policy_per_episode.data

        for summarization_key in summarization_keys:
            if summarization_key not in summarization_per_policy.keys():
                summarization_per_policy[summarization_key] = []

        summarization_per_policy["episode_rewards"] += [
            np.sum(sample_batch_per_policy_per_episode["rewards"])]
        summarization_per_policy["episode_lengths"] += [
            np.shape(sample_batch_per_policy_per_episode["rewards"])[0]]

    for summarization_key in summarization_keys:

        summarization_per_policy["{}_mean".format(summarization_key)] = np.mean(
            summarization_per_policy[summarization_key])
        summarization_per_policy["{}_max".format(summarization_key)] = np.max(
            summarization_per_policy[summarization_key])
        summarization_per_policy["{}_min".format(summarization_key)] = np.min(
            summarization_per_policy[summarization_key])

    return summarization_per_policy


def override_dict(dict_, args):
    """Override dict_ with the items in args.
    If items in dict_ is not presented in args, keep it as it is.
    """
    if isinstance(dict_, dict) and isinstance(args, dict):
        dict_ = dcopy(dict_)
        for key, value in args.items():
            dict_[key] = value
        return dict_
    else:
        raise TypeError


def is_gridsearch_match(config, value):
    """Check if a config match a value or
    (if it is a gridsearch) if it contains only one item and the item matches the value
    """
    if is_grid_search(config):
        return is_list_match(config["grid_search"], value)
    else:
        if config == value:
            return True
        else:
            return False


def is_list_match(item, value):
    """Check if a item match a value or
    (if it is a list) if it contains only one item and the item matches the value
    """
    if isinstance(item, list):
        if len(item) == 1 and item[0] == value:
            return True
        else:
            return False
    else:
        if item == value:
            return True
        else:
            return False


def get_list_from_gridsearch(config, enable_config=True, default=None):
    """Get a list from a config that could be a grid_search.
    If it is a grid_search, return a list of the configs.
    If not, return the single config, but in a list.
    If not enable_config, return default.
    """
    config = dcopy(config)
    if enable_config:
        if is_grid_search(config):
            return config["grid_search"]
        else:
            return [config]
    else:
        return [default]


def get_one_from_grid_search(config, index=0):
    """Get one of the configs in a config that is a grid_search.
    If it is not a grid_search, return it as it is.
    """
    config = dcopy(config)
    if is_grid_search(config):
        return config["grid_search"][index]
    else:
        return config


def is_grid_search(config):
    """Check of a config is a grid_search.
    """
    return isinstance(config, dict) and len(config.keys()) == 1 and list(config.keys())[0] == "grid_search"


def get_social_config(env):
    """
    Arguments:
        env: Arena-Blowblow-Sparse-2T2P-Discrete

    Returns:
        [[0,1],[2,3]]
    """

    xTxP = env.split("-")[3]

    T = int(xTxP.split("T")[0])
    P = int(xTxP.split("T")[1].split("P")[0])

    policy_i = 0
    all_list = []
    for t in range(T):
        t_list = []
        for p in range(P):
            t_list += [copy.deepcopy(policy_i)]
            policy_i += 1
        all_list += [copy.deepcopy(t_list)]

    return all_list


def human_select(choices, prefix_msg="", key="unamed", default_index=-1):

    choices = list(choices)

    if len(choices) < 1:
        raise ValueError

    for i in range(len(choices)):
        choices[i] = str(choices[i])

    for i in range(len(choices)):
        choices[i] = {
            'name': choices[i]
        }

    choices[default_index]['checked'] = True

    questions = []
    question = {
        'type': 'checkbox',
        'qmark': '*',
        'message': "WARNING: {} There are multiple {} as follows:".format(
            prefix_msg,
            key,
        ),
        'name': key,
        'choices': choices,
        'validate': lambda answer: 'You must choose at least one of them.'
        if len(answer) == 0 else True,
        # 'pointer_index': default_index if default_index > 0 else (len(choices) + default_index),
    }
    questions += [question]

    answers = prompt(questions, style=custom_style_2)

    return answers[key][0]


def list_to_selection_dict(list_):
    """
    Example:
        Arguments:
            list_: ["a","b"]
        Returns:
            {
                0: "a",
                1: "b",
            }
    """
    selection_dict = {}
    for i in range(len(list_)):
        selection_dict[i] = list_[i]
    return selection_dict


def prepare_path(path):
    """Check if path exists, if not, create one.
    """
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(
                os.path.dirname(
                    path
                )
            )
        except OSError as exc:
            # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def remove_repeats_in_list(list_):
    """Remove repeats in a list.
    """
    return list(dict.fromkeys(list_).keys())


def list_subtract(x, y):
    """Substract list y from list x.
    """
    return [item for item in x if item not in y]


def to_dir_str(str_):
    """Convert a str to a str that can be used as a dir path. Specifically,

        * remove spaces
        * remove '
        * replace [] with ()
    """
    str_ = str_.replace(" ", "")
    str_ = str_.replace("'", "")
    str_ = str_.replace("[", "(")
    str_ = str_.replace("]", ")")
    return str_


def update_config_value_by_key_value(config_to_update, config_key, config_value):
    """Update config_to_update at config_key to config_value

    Example:
        Arguments:
            config_to_update: {
                "a": "b",
                "c": {
                    "d": "e",
                    "f": "g",
                },
            }
            config_key: "c-f"
            config_value: "h"
        Returns:
            {
                "a": "b",
                "c": {
                    "d": "e",
                    "f": "h",
                },
            }
    """
    temp = config_to_update
    config_key = config_key.split("-")
    len_config_key = len(config_key)
    for i in range(len_config_key):
        if i < (len_config_key - 1):
            temp = temp[config_key[i]]
        elif i == (len_config_key - 1):
            temp[config_key[i]] = config_value
        else:
            raise ValueError
    return config_to_update


def get_config_value_by_key(config_to_get, config_key):
    """Update config_to_update at config_key to config_value

    Example:
        Arguments:
            config_to_update: {
                "a": "b",
                "c": {
                    "d": "e",
                    "f": "g",
                },
            }
            config_key: "c-f"
        Returns:
            "g"
    """
    temp = config_to_get
    config_key = config_key.split("-")
    len_config_key = len(config_key)
    for i in range(len_config_key):
        if i < (len_config_key - 1):
            temp = temp[config_key[i]]
        elif i == (len_config_key - 1):
            return temp[config_key[i]]
        else:
            raise ValueError


def update_config_value_by_config(config_to_update, config):
    """Update config_to_update according to config

    Example:
        Arguments:
            config_to_update: {
                "a": "b",
                "c": {
                    "d": "e",
                    "f": "g",
                },
            }
            config: {
                "c-f": "h",
                "c-d": "g",
            }
        Returns:
            {
                "a": "b",
                "c": {
                    "d": "g",
                    "f": "h",
                },
            }
    """
    for config_key, config_value in config.items():
        update_config_value_by_key_value(
            config_to_update=config_to_update,
            config_key=config_key,
            config_value=config_value,
        )
    return config_to_update


def simplify_config_key(config_key):
    """
    Example:
        Arguments:
            config_key: "config-env_config-is_shuffle_agents"
        Returns:
            "c-ec-isa"
    """
    config_key = config_key.replace("-", " - ")
    config_key = config_key.replace("_", " ")
    words = config_key.split()
    letters = [word[0] for word in words]
    return_ = "".join(letters)
    return return_


def running_config_to_str(running_config):
    """
    Example:
        Arguments:
            running_config: {'env': 'Arena-Tennis-Sparse-2T1P-Discrete', 'config-num_learning_policies': 1}
        Returns:
            "e=Arena-Tennis-Sparse-2T1P-Discrete,c-nlp=1"
    """
    str_ = ""
    for running_config_key, running_config_value in running_config.items():
        str_ += ",{}={}".format(
            simplify_config_key(str(running_config_key)),
            running_config_value,
        )
    return str_


def find_in_list_of_list(list_, item):
    """
    Example:
        Arguments:
            list_: [[0,1],[2,3]]
            item: 2
        Returns:
            (1,0)
    """
    for sub_list in list_:
        if item in sub_list:
            return (list_.index(sub_list), sub_list.index(item))
    raise ValueError("'{}' is not in list".format(item))


def flatten_list(list_):
    flat_list = []
    for sublist in list_:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def try_reduce_list(list_):
    if len(list_) == 1:
        return list_[0]
    else:
        return list_


def try_reduce_dict(dict_):
    if len(dict_.values()) == 1:
        return list(dict_.values())[0]
    else:
        return dict_


def get_key_in_parse_from_config_key(config_key):
    """
    Example:
        Arguments:
            config_key: "a-b-c"
        Returns:
            "c"
    """
    return config_key.split("-")[-1]


def get_shared_scope(share_layer_policies, policy_i):
    """
    Example:
        Arguments:
            share_layer_policies: [[0,1],[2,3]]
            policy_i: 2
        Returns:
            [2,3]
    """
    return dcopy(
        share_layer_policies[
            find_in_list_of_list(
                share_layer_policies, policy_i
            )[0]
        ]
    )


def replace_in_tuple(tup, index, value):
    lst = list(tup)
    lst[index] = value
    tup = tuple(lst)
    return tup


def get_img_from_fig(fig, dpi=180):
    # define a function which returns an image as numpy array from figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_feature(data, label=None, y_range=None, new_fig=True, fig=None):
    # plot a feature of size(x)
    if new_fig:
        fig = plt.figure()
    ax = plt.gca()
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.plot(np.arange(np.shape(data)[0]), data, label=label)
    if label is not None:
        ax.legend()
    if new_fig:
        plt.close()
    return fig
