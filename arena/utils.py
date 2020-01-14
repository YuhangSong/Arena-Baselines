import os
import re
import cv2
import io
import logging
import ray
import platform
import random
import gym

import copy
from copy import deepcopy as dcopy

import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')


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
    if is_grid_search(config):
        return config["grid_search"][index]
    else:
        return config


def is_grid_search(config):
    """Check of a config is a grid_search.
    """
    return isinstance(config, dict) and len(config.keys()) == 1 and list(config.keys())[0] == "grid_search"


def get_social_config(env):

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


def list_subtract(x, y):
    """Substract list y from list x.
    """
    return [item for item in x if item not in y]


def list_to_dict_str(mylist):
    outstr = ''
    for i in range(len(mylist)):
        outstr += mylist[i]
        if i < (len(mylist) - 1):
            outstr += "-"
    return outstr


def find_in_list_of_list(mylist, item):
    for sub_list in mylist:
        if item in sub_list:
            return (mylist.index(sub_list), sub_list.index(item))
    raise ValueError("'{}' is not in list".format(item))


def flatten_list(mylist):
    flat_list = []
    for sublist in mylist:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def try_reduce_list(mylist):
    if len(mylist) == 1:
        return mylist[0]
    else:
        return mylist


def try_reduce_dict(mydict):
    if len(mydict.values()) == 1:
        return list(mydict.values())[0]
    else:
        return mydict


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
