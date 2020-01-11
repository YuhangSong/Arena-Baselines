from .arguments import create_parser, create_experiments

from .arena import ArenaRllibEnv
from .arena import ARENA_ENV_PREFIX, AGENT_ID_PREFIX, POLICY_ID_PREFIX
from .arena import create_arena_experiments
from .arena import get_policy_id, get_agent_i
from .arena import is_arena_env, is_all_arena_env, is_any_arena_env, remove_arena_env_prefix

from .utils import get_list_from_gridsearch, get_one_from_grid_search, is_grid_search
from .utils import prepare_path
from .utils import find_in_list_of_list
from .utils import plot_feature, get_img_from_fig

from ray.tune.registry import register_env

register_env(
    "Arena-Tennis-Sparse-2T1P-Discrete",
    lambda env_config: ArenaRllibEnv(
        "Tennis-Sparse-2T1P-Discrete",
        env_config=env_config,
    ))

register_env(
    "Arena-Blowblow-Sparse-2T2P-Discrete",
    lambda env_config: ArenaRllibEnv(
        "Blowblow-Sparse-2T2P-Discrete",
        env_config=env_config,
    ))
