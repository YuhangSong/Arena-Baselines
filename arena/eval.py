from .utils import *
from .arena import *


def inquire_checkpoints(policy_ids):

    checkpoints = {}
    prefix_msg = 'Setting '

    for policy_id in policy_ids:

        prefix_msg += "{}".format(
            policy_id,
        )

        answers = prompt(
            [
                {
                    'type': 'list',
                    'name': 'policy_id',
                    'message': '{}. Would you like to copy checkpoints config from the following policy_id?'.format(
                        prefix_msg,
                    ),
                    'choices': list(checkpoints.keys()) + ['No, create a new config']
                },
            ],
            style=custom_style_2,
        )

        if answers['policy_id'] in checkpoints.keys():

            checkpoints[policy_id] = dcopy(
                checkpoints[answers['policy_id']])

        else:

            checkpoints[policy_id] = {}

            logdirs = human_select(
                choices=get_possible_logdirs(),
                prefix_msg=prefix_msg,
                key="logdir",
            )

            for logdir in logdirs:

                prefix_msg += ", logdir={}".format(
                    '...' + logdir[-5:],
                )

                answers = prompt(
                    [
                        {
                            'type': 'list',
                            'name': 'logdir',
                            'message': '{}. Would you like to copy checkpoints config from the following logdir?'.format(
                                prefix_msg,
                            ),
                            'choices': list(checkpoints[policy_id].keys()) + ['No, create a new config']
                        },
                    ],
                    style=custom_style_2,
                )

                if answers['logdir'] in checkpoints[policy_id].keys():

                    checkpoints[policy_id][logdir] = dcopy(
                        checkpoints[policy_id][answers['logdir']])

                else:

                    checkpoints[policy_id][logdir] = {}

                    population_is = human_select(
                        choices=get_possible_populations(
                            logdir=logdir
                        ),
                        prefix_msg=prefix_msg,
                        key="population_i",
                    )

                    for population_i in population_is:

                        prefix_msg += ", population_i={}".format(
                            population_i,
                        )

                        answers = prompt(
                            [
                                {
                                    'type': 'list',
                                    'name': 'population_i',
                                    'message': '{}. Would you like to copy checkpoints config from the following population_i?'.format(
                                        prefix_msg,
                                    ),
                                    'choices': list(checkpoints[policy_id][logdir].keys()) + ['No, create a new config']
                                },
                            ],
                            style=custom_style_2,
                        )

                        if answers['population_i'] in checkpoints[policy_id][logdir].keys():

                            checkpoints[policy_id][logdir][population_i] = dcopy(
                                checkpoints[policy_id][logdir][answers['population_i']])

                        else:

                            checkpoints[policy_id][logdir][population_i] = []

                            possible_iteration_indexes, possible_iterations = get_possible_iteration_indexes(
                                logdir=logdir,
                                population_i=population_i,
                            )

                            iteration_indexes = human_select(
                                choices=possible_iteration_indexes,
                                prefix_msg=prefix_msg,
                                key="iteration_index",
                            )

                            for iteration_index in iteration_indexes:
                                checkpoints[policy_id][logdir][population_i] += [
                                    possible_iterations[
                                        int(iteration_index)
                                    ]
                                ]

    return checkpoints


def run_result_matrix(checkpoint_paths, worker, policy_ids=None):

    if policy_ids is None:
        policy_ids = list(checkpoint_paths.keys())

    if len(policy_ids) == 0:

        summarization = summarize_sample_batch(worker.sample())

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
