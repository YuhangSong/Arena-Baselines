ARENA_ENV_PREFIX = 'Arena-'
AGENT_ID_PREFIX = "agent"
POLICY_ID_PREFIX = "policy"
CHECKPOINT_PATH_PREFIX = "learning_agent/"
CHECKPOINT_PATH_POPULATION_PREFIX = "p_"
CHECKPOINT_PATH_ITERATION_PREFIX = "i_"


def policy_i2id(policy_i):
    """Get policy_id from policy_i.
    """
    return "{}_{}".format(POLICY_ID_PREFIX, policy_i)


def agent_id2i(agent_id):
    """Get agent_i from agent_id.
    """
    return int(agent_id.split(AGENT_ID_PREFIX + "_")[1])


def agent_i2id(agent_i):
    """Get agent_id from agent_i.
    """
    return "{}_{}".format(AGENT_ID_PREFIX, agent_i)
