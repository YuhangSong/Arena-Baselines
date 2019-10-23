import multiprocessing as mp
import os
import platform
from .arena_gym_unity import ArenaUnityEnv as UnityEnv
from .arena_subproc_vec_env import ArenaSubprocVecEnv as SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.bench import Monitor
from baselines import logger


def get_env_directory(env_name):
    '''
        get_env_directory:
            get env path according to env_name
    '''
    return os.path.join(
        os.path.dirname(__file__),
        'bin/{}-{}'.format(
            env_name,
            platform.system(),
        )
    )


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def arena_make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, use_visual=True):  # pylint: disable=C0111
        def _thunk():
            # arena-spec: multiagent=True
            env = UnityEnv(env_directory, rank, use_visual=use_visual,
                           uint8_visual=True, multiagent=True)
            # arena-spec
            env.set_train_mode(train_mode=True)
            # arena-spec: remove Monitor as it does not support multiagent
            # env = Monitor(env, logger.get_dir() and os.path.join(
            #     logger.get_dir(), str(rank)))
            return env
        return _thunk
    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])
