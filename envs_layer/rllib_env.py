from ray.rllib.env.multi_agent_env import MultiAgentEnv

ROCK = 0
PAPER = 1
SCISSORS = 2


class ArenaRllibEnv(MultiAgentEnv):
    """Two-player environment for rock paper scissors.

    The observation is simply the last opponent action."""

    def __init__(self, _):
        self.action_space = Discrete(3)
        self.observation_space = Discrete(3)
        self.player1 = "player1"
        self.player2 = "player2"
        self.last_move = None
        self.num_moves = 0

    def reset(self):
        self.last_move = (0, 0)
        self.num_moves = 0
        return {
            self.player1: self.last_move[1],
            self.player2: self.last_move[0],
        }

    def step(self, action_dict):
        move1 = action_dict[self.player1]
        move2 = action_dict[self.player2]
        self.last_move = (move1, move2)
        obs = {
            self.player1: self.last_move[1],
            self.player2: self.last_move[0],
        }
        r1, r2 = {
            (ROCK, ROCK): (0, 0),
            (ROCK, PAPER): (-1, 1),
            (ROCK, SCISSORS): (1, -1),
            (PAPER, ROCK): (1, -1),
            (PAPER, PAPER): (0, 0),
            (PAPER, SCISSORS): (-1, 1),
            (SCISSORS, ROCK): (-1, 1),
            (SCISSORS, PAPER): (1, -1),
            (SCISSORS, SCISSORS): (0, 0),
        }[move1, move2]
        rew = {
            self.player1: r1,
            self.player2: r2,
        }
        self.num_moves += 1
        done = {
            "__all__": self.num_moves >= 10,
        }
        return obs, rew, done, {}
