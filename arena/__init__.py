from .arena import *
from .eval import *
from .vis import *
from .utils import prepare_path
from .rollout_worker import ArenaRolloutWorker

from ray.tune.registry import register_env

game_files = [
    "Arena-Tennis-Sparse-2T1P-Discrete",
    "Arena-BarrierGunner-3X3-PT-Sparse-2T1P-Discrete",
]

for game_file in game_files:
    register_env(
        game_file,
        lambda env_config: ArenaRllibEnv(
            game_file,
            env_config=env_config,
        ))
