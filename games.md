## Games

### [Tennis](https://youtu.be/)

![Tennis](images/Tennis.png)

* Set-up: Two-player game where agents control rackets to bounce ball over a net.
* Variants: Tennis-Sparse-2T1P-{Discrete,Continuous}
* Goal: The agents must not let the ball touch the ground of their own side.
* Agents: The environment contains two agents with same Behavior Parameters.
* Agent Reward Function:
  * +0.0 To the agent whose ground is touched by the ball.
  * +1.0 To the other agent.
* Episode Terminate Condition:
  * The ball touched the ground.
* Behavior Parameters:
  * Vector Observations:
    * Lidar
  * Vector Action space:
    * Discrete: [General Player Discrete](##-general-player-discrete).
    * Continuous: [General Player Continuous](##-general-player-continuous).
  * Visual Observations:
    * Visual_FP
    * Visual_FP
* Reset Parameters: Three
    * angle: Angle of the racket from the vertical (Y) axis.
      * Default: 55
      * Recommended Minimum: 35
      * Recommended Maximum: 65
    * gravity: Magnitude of gravity
      * Default: 9.81
      * Recommended Minimum: 6
      * Recommended Maximum: 20
    * scale: Specifies the scale of the ball in the 3 dimensions (equal across the three dimensions)
      * Default: 1
      * Recommended Minimum: 0.2
      * Recommended Maximum: 5
* Benchmark Mean Reward: 2.5
* Config
  * ```python train.py -f ./arena-experiments/Benchmark-2T1P-Discrete.yaml```
* restore
  * ```python train.py -f ./arena-experiments/Benchmark-2T1P-Discrete.yaml --resume```


## Common Configs

### Action Space

#### General Player Discrete

* 0: Nope action
* 1:

#### General Player Continuous

* 0: Nope action
* 1:
