def create_parser():
    """Returns parser with additional arena configs.
    """

    # import parser from rllib.train
    from ray.rllib.train import create_parser as create_parser_rllib
    parser = create_parser_rllib()

    parser.add_argument(
        "--is-shuffle-agents",
        action="store_true",
        help=(
            "Whether shuffle agents every episode. "
            "This helps the trained policies to have better generalization ability. "
        ))

    parser.add_argument(
        "--train-mode",
        action="store_false",
        default=True,
        help=(
            "Whether run Arena environments in train mode. "
            "In train mode, the Arena environments run in a faster clock and in smaller resulotion. "
        ))

    parser.add_argument(
        "--obs-type",
        default="visual_FP",
        type=str,
        help=(
            "Type of the observation. Options are as follows: "
            "vector (low-dimensional vector observation); "
            "visual_FP (first-person visual observation); "
            "visual_TP (third-person visual observation); "
            "obs1-obs2-... (combine multiple types of observations); "
        ))

    parser.add_argument(
        "--iterations-per-reload",
        default=1,
        type=int,
        help=(
            "Number of iterations between each reload. "
            "In each reload, learning policies are saved and all policies are reloaded. "
        ))

    parser.add_argument(
        "--num-learning-policies",
        default="independent",
        type=str,
        help=(
            "How many agents in the game are bound to learning policies (one to each). Options are as follows: "
            "all (all agents are bound to learning policies, one for each. This is also known as independent learner.); "
            "x (there are x agents bound to x learning policies, one for each; the other (num_agents-x) agents are bound to playing policies, one for each.); "
            "Setting x=1 is known as selfplay. "
            "Playing policies donot explore or update, but they keep reloading weights from the current and previous learning policy at each reload. "
        ))

    parser.add_argument(
        "--playing-policy-load-recent-prob",
        default=0.8,
        type=float,
        help=(
            "When reload, for playing policies only, the probability of chosing recent learning policy, against chosing uniformly among historical ones. "
        ))

    parser.add_argument(
        "--size-population",
        default=1,
        type=int,
        help=(
            "Number of policies to be trained in population-based training. "
            "In each reload, each one of all learning/player policies will be reloaded with one of the size_population policies randomly. "
        ))

    parser.add_argument(
        "--share-layer-policies",
        default=None,
        help=(
            "Specify the policies that share layers. Options are as follows: "
            "none; "
            "team; "
            "[[a,b,c,...],[x,y,z,...],...] (policies of id a,b,c,... will share layers, policies of id x,y,z,... will share layers, ...); "
        ))

    return parser


def create_experiments(args):
    """Create configs from args
    """

    input("# WARNING: it is recommended to use -f CONFIG.yaml, instead of passing args. Press hit enter to continue. ")

    # Note: keep this in sync with tune/config_parser.py
    experiments = {
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
                    obs_type=args.obs_type,
                ),
                iterations_per_reload=args.iterations_per_reload,
                num_learning_policies=args.num_learning_policies,
                playing_policy_load_recent_prob=args.playing_policy_load_recent_prob,
                size_population=args.size_population,
                share_layer_policies=args.share_layer_policies,
            ),
            "restore": args.restore,
            "num_samples": args.num_samples,
            "upload_dir": args.upload_dir,
        }
    }

    return experiments
