from .utils import *
from .arena import *


def inquire_checkpoints(local_dir, policy_ids):
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

            logdirs = inquire_select(
                choices=get_possible_logdirs(local_dir),
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

                    population_is = inquire_select(
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

                            iteration_indexes = inquire_select(
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


def run_result_matrix(checkpoint_paths, worker, policy_ids=None, policy_loading_status={}, checkpoint_path_abbreviated_to=68):
    """
    Arguments:
        checkpoint_paths:
        worker:
        policy_ids:
    Returns:
        result_matrix: see https://github.com/YuhangSong/Arena-Baselines/#evaluate-and-visualize-evaluation
    """

    if policy_ids is None:
        policy_ids = list(checkpoint_paths.keys())

    if len(policy_ids) == 0:

        print("============================= sampling... =============================")

        sampled = worker.sample()
        worker.env.reset()

        summarization = summarize_sample_batch(sampled)

        print("policy_loading_status:")
        print(summarize(policy_loading_status))
        print("summarization:")
        print(summarize(summarization))

        result_matrix = []
        for policy_id in summarization.keys():
            result_matrix += [
                summarization[policy_id]['episode_rewards_mean']
            ]

        return result_matrix

    else:

        policy_id = policy_ids[0]

        result_matrix = []

        for checkpoint_path_i, checkpoint_path in enumerate(checkpoint_paths[policy_id]):

            if checkpoint_path_abbreviated_to > 0:
                checkpoint_path_abbreviated = "...{}".format(
                    checkpoint_path[-checkpoint_path_abbreviated_to:]
                )
            else:
                checkpoint_path_abbreviated = checkpoint_path

            policy_loading_status[policy_id] = {
                "checkpoint_path_i": checkpoint_path_i,
                "checkpoint_path": checkpoint_path_abbreviated,
            }

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
                        policy_loading_status=policy_loading_status,
                    )
                )
            )

        return result_matrix
