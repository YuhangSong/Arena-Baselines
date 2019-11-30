from .rllib_env import ArenaRllibEnv
from ray.tune.registry import register_env

ARENA_ENV_PREFIX = 'Arena-'


def get_one_from_grid_search(config, index=0):
    if is_grid_search(config):
        return config["grid_search"][index]
    else:
        return config


def is_grid_search(config):
    return isinstance(config, dict) and len(config.keys()) == 1 and config.keys()[0] == "grid_search"


def is_arena_env(each_env):
    return each_env[:len(ARENA_ENV_PREFIX)] == ARENA_ENV_PREFIX


def is_all_arena_env(env):
    if is_grid_search(env):
        for each_env in env["grid_search"]:
            if not is_arena_env(each_env):
                return False
        return True
    else:
        return is_arena_env(env)


def is_any_arena_env(env):
    if is_grid_search(env):
        for each_env in env["grid_search"]:
            if is_arena_env(each_env):
                return True
        return False
    else:
        return is_arena_env(env)


def remove_arena_env_prefix(env):
    if is_all_arena_env(env):
        if is_grid_search(env):
            for each_env in env["grid_search"]:
                each_env = each_env[len(ARENA_ENV_PREFIX):]
    else:
        raise NotImplementedError


register_env(
    "Arena-Tennis-Sparse-2T1P-Discrete",
    lambda env_config: ArenaRllibEnv(
        "Tennis-Sparse-2T1P-Discrete",
        env_config=env_config,
    ))
