import copy

from .arena import ARENA_ENV_PREFIX, AGENT_ID_PREFIX, POLICY_ID_PREFIX


def get_policy_id(policy_i):
    """Get policy_id from policy_i.
    """
    return "{}_{}".format(POLICY_ID_PREFIX, policy_i)


def get_agent_i(agent_id):
    """Get agent_i from agent_id.
    """
    return int(agent_id.split(AGENT_ID_PREFIX + "_")[1])


def policy_mapping_fn_i2i(agent_id):
    """A policy_mapping_fn that maps agent i to policy i.
    """
    return get_policy_id(get_agent_i(agent_id))


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


def is_arena_env(each_env):
    """Check if a env (string) is an arena env.
    """
    return each_env[:len(ARENA_ENV_PREFIX)] == ARENA_ENV_PREFIX


def is_all_arena_env(env):
    """Check if all env in a grid_search env are arena env.
    If env is not a grid_search, return is_arena_env.
    """
    if is_grid_search(env):
        for each_env in env["grid_search"]:
            if not is_arena_env(each_env):
                return False
        return True
    else:
        return is_arena_env(env)


def is_any_arena_env(env):
    """Check if any env in a grid_search env is arena env.
    If env is not a grid_search, return is_arena_env.
    """
    if is_grid_search(env):
        for each_env in env["grid_search"]:
            if is_arena_env(each_env):
                return True
        return False
    else:
        return is_arena_env(env)


def remove_arena_env_prefix(env):
    """Remove ARENA_ENV_PREFIX from a env (possibly a grid_search).
    """
    env = copy.deepcopy(env)
    if is_grid_search(env):
        if is_all_arena_env(env):
            for i in range(len(env["grid_search"])):
                env["grid_search"][i] = remove_arena_env_prefix(
                    env["grid_search"][i]
                )
            return env
        else:
            raise NotImplementedError
    else:
        return env[len(ARENA_ENV_PREFIX):]
