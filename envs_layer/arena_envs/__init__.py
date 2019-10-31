import os
import platform
from .arena_gym_unity import ArenaUnityEnv as UnityEnv


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
