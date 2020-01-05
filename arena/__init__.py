from .arena import ArenaRllibEnv
from .arena import ARENA_ENV_PREFIX, AGENT_ID_PREFIX, POLICY_ID_PREFIX
from .utils import get_policy_id, get_agent_i, policy_mapping_fn_i2i, get_list_from_gridsearch, get_one_from_grid_search, is_grid_search, is_arena_env, is_all_arena_env, is_any_arena_env, remove_arena_env_prefix
from .utils import DeterministicCategorical

from ray.tune.registry import register_env


register_env(
    "Arena-Tennis-Sparse-2T1P-Discrete",
    lambda env_config: ArenaRllibEnv(
        "Tennis-Sparse-2T1P-Discrete",
        env_config=env_config,
    ))
