import logging

from .utils import *

logger = logging.getLogger(__name__)


def create_parser():
    """Returns parser with additional arena configs.
    """

    # import parser from rllib.train
    from ray.rllib.train import create_parser as create_parser_rllib
    parser = create_parser_rllib()

    parser.add_argument(
        "--is-shuffle-agents",
        action="store_false",
        default=True,
        help=(
            "Whether shuffle agents every episode. "
            "This helps the trained policies to have better generalization ability. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--train-mode",
        action="store_false",
        default=True,
        help=(
            "Whether run Arena environments in train mode. "
            "In train mode, the Arena environments run in a faster clock and in smaller resulotion. "
            "This config does not support grid_search. "
        ))

    parser.add_argument(
        "--sensors",
        default=["vector"],
        help=(
            "Type of the observation. Options are as follows: "
            "[vector] (low-dimensional vector observation); "
            "[visual_FP] (first-person visual observation); "
            "[visual_TP] (third-person visual observation); "
            "[xx_sensor, yy_sensor, ...] (combine multiple types of observations, the observation_space would be gym.spaces.Dict and the returned observation per agent is a dict, where keys are xx_multi_agent_obs-xx_sensor); "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--multi-agent-obs",
        default=["own"],
        help=(
            "For Arena multi-agent environments, which observation to use. Options are as follows: "
            "[own] (the agent's own observation); "
            "[team_absolute] (the team's observations, the position of own observation is absolute); "
            "[team_relative] (the team's observations, the position of own observation is relative); "
            "[all_absolute] (all agents' observations, the position of own and team observations are absolute); "
            "[all_relative] (all agents' observations, the position of own and team observations are relative); "
            "[xx_multi_agent_obs, yy_multi_agent_obs, ...] (combine multiple types of observations, the observation_space would be gym.spaces.Dict and the returned observation per agent is a dict, where keys are xx_multi_agent_obs-xx_sensor); "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--iterations-per-reload",
        default=1,
        help=(
            "Number of iterations between each reload. "
            "In each reload, learning policies are saved and all policies are reloaded. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--num-learning-policies",
        default=1,
        help=(
            "How many agents in the game are bound to learning policies (one to each). Options are as follows: "
            "all (all agents are bound to learning policies, one for each. This is also known as independent learner.); "
            "x (there are x agents bound to x learning policies, one for each; the other (num_agents-x) agents are bound to playing policies, one for each.); "
            "Setting x=1 is known as selfplay. "
            "Playing policies donot explore or update, but they keep reloading weights from the current and previous learning policy at each reload. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--playing-policy-load-recent-prob",
        default=0.8,
        help=(
            "When reload, for playing policies only, the probability of chosing recent learning policy, against chosing uniformly among historical ones. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--size-population",
        default=1,
        help=(
            "Number of policies to be trained in population-based training. "
            "In each reload, each one of all learning/player policies will be reloaded with one of the size_population policies randomly. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--share-layer-policies",
        default=[],
        help=(
            "Specify the policies that share layers. Options are as follows: "
            "[]; "
            "team; "
            "[[a,b,c,...],[x,y,z,...],...] (policies of id a,b,c,... will share layers, policies of id x,y,z,... will share layers, ...); "
            "After setting this up, you additionally need to go to arena.models.ArenaPolicy, defining which layers you want to share across the defined scope. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--actor-critic-obs",
        default=[],
        help=(
            "Specify the observations of actor and critic separately. Options are as follows: "
            "[] (not taking effect); "
            "[xx, yy] (actor will use xx as observations, critic will use yy as observations); "
            "xx and yy should be one of the items in the config of sensors. "
            "If xx or yy are not in sensors, the config of sensors will be overrided to include xx and yy. "
            "This config supports grid_search. "
        ))

    parser.add_argument(
        "--dummy",
        action="store_true",
        default=False,
        help=(
            "For debug, wheter run in dummy mode, which requires minimal resources. "
            "This config does not support grid_search. "
        ))

    return parser


def varify_actor_critic_obs(actor_critic_obs):
    """Verify that actor_critic_obs is valid.
    """
    if not((len(actor_critic_obs) == 0) or (len(actor_critic_obs) == 2)):
        raise Exception(
            "actor_critic_obs can only be [] or [xx, yy]")


def override_exps_according_to_dummy(exps, dummy):
    """Overide exps according to dummy.
    """
    exps = dcopy(exps)
    if dummy:
        logger.warning(
            "Run in dummy mode. "
            "Overriding configs. "
        )
        for exp_key in exps.keys():
            exps[exp_key]["config"]["num_gpus"] = 0
            exps[exp_key]["config"]["num_workers"] = 1
            exps[exp_key]["config"]["num_envs_per_worker"] = 1
            exps[exp_key]["config"]["sample_batch_size"] = 100
            exps[exp_key]["config"]["train_batch_size"] = 100
            exps[exp_key]["config"]["sgd_minibatch_size"] = 100
    return exps


def preprocess_num_learning_policies_keys(num_learning_policies_keys, number_agents):
    """
    """
    for i in range(len(num_learning_policies_keys)):
        if isinstance(num_learning_policies_keys[i], str):
            if num_learning_policies_keys[i] == "all":
                num_learning_policies_keys[i] = dcopy(number_agents)
    return remove_repeats_in_list(num_learning_policies_keys)


def preprocess_share_layer_policies_keys(share_layer_policies_keys, env):
    """
    """
    for i in range(len(share_layer_policies_keys)):
        if isinstance(share_layer_policies_keys[i], str):
            if share_layer_policies_keys[i] == "team":
                share_layer_policies_keys[i] = dcopy(
                    get_social_config(
                        env
                    )
                )
    return share_layer_policies_keys
    # return remove_repeats_in_list(share_layer_policies_keys)


def preprocess_multi_agent_obs_keys(multi_agent_obs_keys, actor_critic_obs):
    """
    If xx or yy of actor_critic_obs are not in multi_agent_obs, multi_agent_obs will be overrided to include them.
    """
    for i in range(len(multi_agent_obs_keys)):
        for each_actor_critic_obs in actor_critic_obs:
            if each_actor_critic_obs not in multi_agent_obs_keys[i]:
                multi_agent_obs_keys[i] += [each_actor_critic_obs]
                logger.warning(
                    "{} in actor_critic_obs is not in multi_agent_obs, override multi_agent_obs to {}".format(
                        each_actor_critic_obs,
                        multi_agent_obs_keys[i]
                    )
                )
    # return remove_repeats_in_list(multi_agent_obs_keys)
    return multi_agent_obs_keys


def preprocess_is_shuffle_agents_keys(is_shuffle_agents_keys, share_layer_policies):
    is_shuffle_agents_keys = dcopy(is_shuffle_agents_keys)
    for i in range(len(is_shuffle_agents_keys)):
        if share_layer_policies != []:
            if is_shuffle_agents_keys[i] != False:
                is_shuffle_agents_keys[i] = False
                logger.warning(
                    "share_layer_policies is not [], override is_shuffle_agents to {}. ".format(
                        is_shuffle_agents_keys[i]
                    )
                )
    return remove_repeats_in_list(is_shuffle_agents_keys)


def create_exps(args):
    """Create configs from args
    """

    input("# WARNING: it is recommended to use -f CONFIG.yaml, instead of passing args. Press hit enter to continue. ")

    # Note: keep this in sync with tune/config_parser.py
    exps = {
        args.experiment_name: {  # i.e. log to ~/ray_results/default
            "run": args.run,
            "checkpoint_freq": args.checkpoint_freq,
            "keep_checkpoints_num": args.keep_checkpoints_num,
            "checkpoint_score_attr": args.checkpoint_score_attr,
            "local_dir": args.local_dir,
            "resources_per_trial": (
                args.resources_per_trial and
                resources_to_json(args.resources_per_trial)
            ),
            "stop": args.stop,
            "config": dict(
                args.config,
                env=args.env,
                env_config=dict(
                    is_shuffle_agents=args.is_shuffle_agents,
                    train_mode=args.train_mode,
                    sensors=args.sensors,
                    multi_agent_obs=args.multi_agent_obs,

                ),
                iterations_per_reload=args.iterations_per_reload,
                num_learning_policies=args.num_learning_policies,
                playing_policy_load_recent_prob=args.playing_policy_load_recent_prob,
                size_population=args.size_population,
                share_layer_policies=args.share_layer_policies,
                actor_critic_obs=args.actor_critic_obs,
            ),
            "restore": args.restore,
            "num_samples": args.num_samples,
            "upload_dir": args.upload_dir,
        }
    }

    return exps
