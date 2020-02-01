from .arena import *
from .eval import *
from .vis import *
from .utils import prepare_path
from .rollout_worker import ArenaRolloutWorker

from ray.tune.registry import register_env

env_ids = [
    "Arena-Tennis-Sparse-2T1P-Discrete",
    "Arena-BarrierGunner-3X3-PT-Sparse-2T1P-Discrete",
]

for env_id in env_ids:
    register_env(
        env_id,
        lambda env_config: ArenaRllibEnv(
            env_id,
            env_config=env_config,
        ))
