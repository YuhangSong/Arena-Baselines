from .rllib_env import ArenaRllibEnv
from ray.tune.registry import register_env
register_env("arena_env",
             lambda env_config: ArenaRllibEnv(env_config))
