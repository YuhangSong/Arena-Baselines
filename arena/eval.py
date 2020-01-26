from .utils import *
from .arena import *


def inquire_checkpoints(policy_ids):
    """Promote a series of inquires to get checkpoints.
    Arguments:
        policy_ids: the policy_ids to inquire
    Returns:
        Example: {
            policy_0:
                logdir_0:
                    population_0:
                        [
                            iteration_0,
                            iteration_1,
                            ...,
                        ]
                    population_1:
                        ...,
                logdir_1:
                    ...,
            policy_1:
                ...,
        }
    """

    checkpoints = {}
    prefix_msg = 'Setting '

    for policy_id in policy_ids:

        prefix_msg += "{}".format(
            policy_id,
        )

        answers = prompt(
            [{
                'type': 'list',
                'name': 'policy_id',
                'message': '{}. Would you like to copy config of checkpoints from that of the following policy_id?'.format(
                    prefix_msg,
                ),
                'choices': list(checkpoints.keys()) + ['no, create a new one']
            }],
            style=custom_style_2,
        )

        if answers['policy_id'] in checkpoints.keys():

            checkpoints[policy_id] = dcopy(
                checkpoints[answers['policy_id']]
            )

        else:

            checkpoints[policy_id] = {}

            logdirs = human_select(
                choices=get_possible_logdirs(),
                prefix_msg=prefix_msg,
                name="logdir",
            )

            for logdir in logdirs:

                prefix_msg += ", logdir={}".format(
                    '...' + logdir.split("_")[-1],
                )

                answers = prompt(
                    [{
                        'type': 'list',
                        'name': 'logdir',
                        'message': '{}. Would you like to copy config of checkpoints from that of the following logdir?'.format(
                            prefix_msg,
                        ),
                        'choices': list(checkpoints[policy_id].keys()) + ['no, create a new one']
                    }],
                    style=custom_style_2,
                )

                if answers['logdir'] in checkpoints[policy_id].keys():

                    checkpoints[policy_id][logdir] = dcopy(
                        checkpoints[policy_id][answers['logdir']]
                    )

                else:

                    checkpoints[policy_id][logdir] = {}

                    population_is = human_select(
                        choices=get_possible_populations(
                            logdir=logdir
                        ),
                        prefix_msg=prefix_msg,
                        name="population_i",
                    )

                    for population_i in population_is:

                        prefix_msg += ", population_i={}".format(
                            population_i,
                        )

                        answers = prompt(
                            [{
                                'type': 'list',
                                'name': 'population_i',
                                'message': '{}. Would you like to copy config of checkpoints from that of the following population_i?'.format(
                                    prefix_msg,
                                ),
                                'choices': list(checkpoints[policy_id][logdir].keys()) + ['no, create a new one']
                            }],
                            style=custom_style_2,
                        )

                        if answers['population_i'] in checkpoints[policy_id][logdir].keys():

                            checkpoints[policy_id][logdir][population_i] = dcopy(
                                checkpoints[policy_id][logdir][answers['population_i']]
                            )

                        else:

                            checkpoints[policy_id][logdir][population_i] = []

                            possible_iteration_indexes, possible_iterations = get_possible_iteration_indexes(
                                logdir=logdir,
                                population_i=population_i,
                            )

                            answers = prompt(
                                [{
                                    'type': 'input',
                                    'name': 'step_size',
                                    'message': 'There are {} possible iteration indexes. Enter the step size of skipping:'.format(
                                        len(possible_iteration_indexes)
                                    ),
                                    'default': '1'
                                }],
                                style=custom_style_2,
                            )

                            iteration_indexes = human_select(
                                choices=range(
                                    0,
                                    len(possible_iteration_indexes),
                                    int(answers['step_size'])
                                ),
                                prefix_msg=prefix_msg,
                                name="iteration_index",
                            )

                            for iteration_index in iteration_indexes:
                                checkpoints[policy_id][logdir][population_i] += [
                                    possible_iterations[
                                        int(iteration_index)
                                    ]
                                ]

    return checkpoints


def run_result_matrix(checkpoint_paths, worker, policy_ids=None):
    """
    Arguments:
        checkpoint_paths:
        worker:
        policy_ids:
    Returns:
        result_matrix:
            nested list with the shape of: (
                len(checkpoint_paths[policy_ids[0]]),
                len(checkpoint_paths[policy_ids[1]]),
                ...,
                len(policy_ids)
            )
            Example:
                Value at result_matrix[1,3,0] means the episode_rewards_mean of policy_0
                when load policy_0 with checkpoint_paths[policy_ids[0]][1]
                and load policy_1 with checkpoint_paths[policy_ids[1]][3]
    """

    if policy_ids is None:
        policy_ids = list(checkpoint_paths.keys())

    if len(policy_ids) == 0:

        sampled = worker.sample()
        worker.env.reset()

        summarization = summarize_sample_batch(sampled)

        result_matrix = []
        for policy_id in summarization.keys():
            result_matrix += [
                summarization[policy_id]['episode_rewards_mean']
            ]

        return result_matrix

    else:

        policy_id = policy_ids[0]

        result_matrix = []

        for checkpoint_path in checkpoint_paths[policy_id]:

            worker.policy_map[policy_id].set_weights(
                pickle.load(
                    open(
                        checkpoint_path,
                        "rb"
                    )
                )
            )

            result_matrix.append(
                dcopy(
                    run_result_matrix(
                        checkpoint_paths=checkpoint_paths,
                        worker=worker,
                        policy_ids=policy_ids[1:],
                    )
                )
            )

        return result_matrix
