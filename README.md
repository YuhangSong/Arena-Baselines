<img src="./images/Cover-Landscape-Visual.png" align="middle" width="1000"/>

## Introduction


<!-- | <img src="./images/ArenaCrawlerMove-2T1P-v1-Continuous.gif" align="middle" width="2000"/>  | <img src="./images/ArenaCrawlerPush-2T1P-v1-Continuous.gif" align="middle" width="2000"/>  | <img src="./images/ArenaCrawlerPush-2T2P-v1-Continuous.gif" align="middle" width="2000"/>  | <img src="./images/Crossroads-2T1P-v1-Continuous.gif" align="middle" width="2000"/> |  <img src="./images/Crossroads-2T2P-v1-Continuous.gif" align="middle" width="2000"/> |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| <img src="./images/ArenaCrawlerMove-2T1P-v1-Continuous.png" align="middle" width="2000"/>  | <img src="./images/ArenaCrawlerPush-2T1P-v1-Continuous.png" align="middle" width="2000"/>  | <img src="./images/ArenaCrawlerPush-2T2P-v1-Continuous.png" align="middle" width="2000"/>  | <img src="./images/Crossroads-2T1P-v1-Continuous.png" align="middle" width="2000"/> | <img src="./images/Crossroads-2T2P-v1-Continuous.png" align="middle" width="2000"/> |

| <img src="./images/Soccer-2T1P-v1-Discrete.gif" align="middle" width="2000"/>  | <img src="./images/BlowBlow-2T1P-v1-Discrete.gif" align="middle" width="2000"/>  | <img src="./images/FighterFull-2T1P-v1-Discrete.gif" align="middle" width="2000"/>  | <img src="./images/Crossroads-2T1P-v1-Continuous.gif" align="middle" width="2000"/> |  <img src="./images/Crossroads-2T2P-v1-Continuous.gif" align="middle" width="2000"/> |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| <img src="./images/Soccer-2T1P-v1-Discrete.png" align="middle" width="2000"/>  | <img src="./images/BlowBlow-2T1P-v1-Discrete.png" align="middle" width="2000"/>  | <img src="./images/FighterFull-2T1P-v1-Discrete.png" align="middle" width="2000"/>  | <img src="./images/Crossroads-2T1P-v1-Continuous.png" align="middle" width="2000"/> | <img src="./images/Crossroads-2T2P-v1-Continuous.png" align="middle" width="2000"/> | -->

<img align="left" width="100" height="100" src="./images/Logo.png">

[Arena](https://sites.google.com/view/arena-unity/) is a general evaluation platform and building toolkit for single/multi-agent intelligence.
As a part of [Arena](https://sites.google.com/view/arena-unity/) project, this repository is to provide implementation of state-of-the-art deep multi-agent reinforcement learning baselines.
Arena-Baselines is fully based on [RLlib](https://ray.readthedocs.io/en/latest/rllib.html).
More resources can be found in [Arena Home](https://sites.google.com/view/arena-unity/).
If you use this repository to conduct research, we kindly ask that you [cite the paper](#citation) as a reference.

## Features

### Baselines

10+ most popular deep multi-agent reinforcement learning baselines, including: independent learners, self-play, variable-shared policies (share arbitrary variables/layers), centralized critic via experiences sharing, centralized critic via observations sharing, Q-Mix, arbitrary grouping agents, centralized critic with counterfactual baselines, population-based training, and etc.

| **Baselines** | **Supported** | **Benchmarked** |
| - | -  | - |
| Independent Learners | :heavy_check_mark: | :x: |
| Self-play | :heavy_check_mark: | :x: |
| Population-based Training | :heavy_check_mark: | :x: |
| Share Weights: between Arbitrary Agents / Teams | :heavy_check_mark: | :x: |
| Sharing observations: own, team_absolute, team_relative, all_absolute, all_relative | :heavy_check_mark: | :x: |
| Multiple sensors: first-person visual, second-person visual, vector (lidar), or any combinations | :heavy_check_mark: | :x: |
| Use separated observations for actor and critic (such as centralized critic and decentralized actors). | :heavy_check_mark: | :x: |

### Games

30+ games built by [Arena-BuildingToolkit](https://github.com/YuhangSong/Arena-BuildingToolkit), including Tennis, Tanl, BlowBlow, FallFlat, and etc.
There are also multiple variants of each game with different team setups, social structures, actions spaces, observation spaces (first-person visual, third-person visual, vector or mixed), domain randomizations, and etc.
More documentation of games can be found [here](#games)

| **Games** | **Supported** | **Benchmarked** |
| - | - | - |
| Arena-Tennis-Sparse-2T1P-Discrete | :heavy_check_mark: | :x: |
| Arena-Blowblow-Sparse-2T2P-Discrete | :heavy_check_mark: | :x: |

### Utilities

Many research-oriented utilities, such as:

| **Utilities** | **Supported**(:heavy_check_mark:) / **In-progress**(:heavy_minus_sign:) |
| - | - |
| Tune (grid search parameters, schedule trails, yaml config and etc.) | :heavy_check_mark: |
| Interactively load and evaluate trained policies/populations | :heavy_check_mark: |
| Visualize performance in population | :heavy_minus_sign: |

## Status: Release

We are currently open to any suggestions or pull requests from the community to make Arena better.
To contribute to the project, [joint us in  Slack](https://join.slack.com/t/arena-ml/shared_invite/enQtNjc1NDE1MTY0MjU3LWMxMjZiMTYyNTE3OWIzM2QxZjU5YmI1NTM2YzYzZDZlNjY0NzllMDFlMjA3MGZiN2QxODA1NTJhZDkzOTI3Nzg), and check [To Do list in Trello](https://trello.com/b/zDiljShz).

Part of the project is maintained in a [separated internal repo](https://github.com/YuhangSong/Arena-Baselines-Internal), where we are working on features including:
*
contact us via slack if you want to have access to these features.

## Get Started

### Dependencies

To install above dependencies, run: (we are using specific versions of the dependencies)
```bash
# Set conda source. Only for users behind the Great Wall of China, no need for other users
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
# If you accidentally did above, type: vim ~/.condarc, and reset conda source by removing the first two lines. Finally, run: conda update --all

# Set pip source. Only for users behind the Great Wall of China, no need for other users.
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# If you accidentally did above, reset pip source with: pip config set global.index-url https://pypi.org/simple

# Create a virtual environment
conda remove --name Arena-Baselines --all -y
conda create -n Arena-Baselines python=3.6.9 -y
source activate Arena-Baselines

# Create dir
mkdir Arena
cd Arena

# ML-Agents, only compatible with this checkpoint
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
git checkout 9b1a39982fd03de8f40f85d61f903e6d972fd2cc
cd ml-agents
pip install -e .
cd ..
cd gym-unity
pip install -e .
cd ..
cd ..

# clone code
git clone https://github.com/YuhangSong/Arena-Baselines.git
cd Arena-Baselines

# PyTorch
pip install --upgrade torch torchvision

# TensorFlow GPU
pip install tensorflow-gpu==1.14
# Or TensorFlow CPU
# pip install tensorflow==1.14

# Ray and RLlib
pip install ray[rllib]
pip install ray[debug]
pip install ray==0.7.4

# Other requirements
pip install -r requirements.txt

# ffmpeg is needed for monitor
sudo apt-get install ffmpeg
```

If you are using different versions of ray or tensorflow, finish [run tests](##run-tests) to make sure it works.
Open up a pull request if you find newer versions that also works fine to complete the tests.

If you run into following situations,

* You are running on a remote server without GUI (X-Server).
* Your machine have a X-Server but it does not belongs (started by) your account, so you cannot access it. (If your machine have a X-Server but it belongs (started by) your account, but you cannot use the GUI desktop all the time, see [here](https://github.com/YuhangSong/Arena-Baselines/blob/master/x-server-belongs-to-you.md))

If none of above is your situation, i.e., you are running things on your own desktop, skip this part and go to [Usage](#Usage).
If you are in above situations, follow guidelines [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) to setup a virtual display.
Or you can follow [here](https://github.com/YuhangSong/Arena-Baselines/blob/master/set-up-x-server.md) (This is simpler and get you in place in shorter time, but could be outdated. If so, go to the above link, and consider open a pull requests to give a update of this).

## Usage

First, crate a TMUX session to maintain your process (more necessary if the machine is a server you connect via SSH), and enter virtual environment
```
tmux new-session -s Arena
source activate Arena-Baselines
```

### Run tests

Test your installation of rllib learning environment with Pong (rllib breaks sometimes, so a test is needed before you go further).
In ```./arena-experiments/Arena-Benchmark.yaml```, comment out other envs and leave ```PongNoFrameskip-v4``` there. Then, run:
```
python train.py -f ./arena-experiments/Arena-Benchmark.yaml
```
You should see reward goes up from -21 (as shown in the following), which means you installation works fine.

<img src="./images/Test-Pong.png" align="middle" width="1000"/>

Meet some problems? Open an issue.

Now, recover ```./arena-experiments/Arena-Benchmark.yaml``` you just modified.
Test an Arena environment (Arena could have difficulties lunching due to different reasons, so a test is needed before you go further)
```
python test_arena_rllib_env.py -f ./arena-experiments/Arena-Benchmark.yaml
```
You should see prints like following:
```
INFO:__main__:obs_rllib: {'agent_0': array([0.41156876, ... , 0.23122263]), 'agent_1': array([0.41156882, ... , 0.23122263])}
INFO:__main__:rewards_rllib: {'agent_0': 0.0, 'agent_1': 1.0}
INFO:__main__:dones_rllib: {'agent_0': True, 'agent_1': True, '__all__': True}
INFO:__main__:infos_rllib: {'agent_0': {'text_observation': ['', ''], 'brain_info': <mlagents.envs.brain.BrainInfo object at 0x7f179da92780>, 'shift': 1}, 'agent_1': {'text_observation': ['', ''], 'brain_info': <mlagents.envs.brain.BrainInfo object at 0x7f179da92780>, 'shift': 1}}
episode end, keep going?
```
Hit enter and it keeps rolling.
Meet some problems? Open an issue.

### Reproduce Arena-Benchmark

Now train on Arena games (reproduce our Arena-Benchmark) with:
```
python train.py -f ./arena-experiments/Arena-Benchmark.yaml
```
You should see the ```episode_len_mean``` goes up from 20 (as shown in the following), which means you installation works fine.

<img src="./images/Arena-Benchmark.png" align="middle" width="1000"/>

### Visualization

The code log multiple curves (as well as figures and other formats of data) to help analysis the training process, run:
```
source activate Arena-Baselines && tensorboard --logdir=~/ray_results --port=9999
```
and visit ```http://localhost:9999``` for visualization with tensorboard.

If your port is blocked, use natapp to forward a port:
```
./natapp --authtoken 237e94b5d173a7c3
./natapp --authtoken 09d9c46fedceda3f
```

### Reproduce/resume benchmarks

* To resume/restore a training, append ```--resume``` to the above command in (Reproduce Arena-Benchmark)[##reproduce-arena-benchmark]
  * To log episode video, go to your log dir, find your recent experiment_state file, for example ```experiment_state-2019-11-30_23-17-55.json```. Change the line ```"monitor": false``` to ```"monitor": true```
    * This will take effect until the next time you make changes in your recent (at that time) experiment_state file
* To log episode video in a new training, simply add line ```monitor: True``` in your yaml. For example, see ```"monitor": True``` in ```./arena-experiments/Test-Pong.yaml```

## Documentations

Function ```create_parser``` in ```./train.py``` gives the detailed description of the configs.
Read them to understand the configs.
Note that we do not recommend passing configs via argparse, instead, use yaml file to config you experiment, as what has been done in [run tests](##run-tests).
In this way, you can make sure your experiments are reproducable.

<!-- **Behaviors:**
Set ```--mode vis_train```, so that
* The game engine runs at a real time scale of 1 (when training, it runs 100 times as the real time scale).
* The game runs only one thread.
* The game renders at 1920*1080, where you can observe agents' observations as well as the top-down view of the global state. So you are expected to do this on a desktop with X-Server you can access, instead of [using a remote server](#setup-x-server).
* All agents act deterministically without exploring.
* Two video files (.avi and .gif) of the episode will be saved, so that you can post it on your project website. The orignal resolution is the same as that of your screen, which is 1920*1080 in our case, click on the gif video to see the high-resolution original file. See [here](#introduction).
* A picture (.png) of the episode will be saved, so that you can use it as a visulizatino of your agents' behavior in your paper. The orignal resolution is the same as that of your screen, which is 1920*1080 in our case, click on the image to see the high-resolution original file.  See [here](#introduction).

## More Baselines and Options (Keep Updating)

### Self-play

Above example commands runs a self-play with following options and features:

* In different thread, the agent is taking different roles, so that the learning generalize better.
* ```--reload-playing-agents-principle``` has three options
  * ```recent``` playing agents are loaded with the most recent checkpoint.
	* ```earliest``` playing agents are loaded with the oldest checkpoint. This option is mainly for debugging, since it removes the stochasticity of playing agent.
  * ```uniform``` playing agents are loaded with the a random checkpoint sampled uniformly from all historical checkpoints.
  * ```OpenAIFive``` To avoid strategy collapse, the agent trains 80% of its games against itself and the other 20% against its past selves. Thus, OpenAIFive is 0.8 probability to be recent, 0.2 probability to be uniform.

### Population-based Training

Population-based training is usefull when the game is none-transitive, which means there is no best agent, but there could be a best population.
Population-based training will train a batch of agents instead of just one.
Add argument ```--population-number 32``` to enable population base training, for example:
This will result in training a population of ```32``` agents.

## Benchmarks (Keep Updating)

| Games | Visualization | Game Description | Agents Behavior |
| ------------- | ------------- | ------------- | ------------- |
| Servers: Wx0, Wx1, H4n, W4n, W2n, W5n |
| ArenaCrawler-Example-v2-Continuous | <img src="./images/ArenaCrawler-Example-v2-Continuous.gif" align="middle" width="2000"/> |  | Server: W4n running. |
| ArenaCrawlerMove-2T1P-v1-Continuous | <img src="./images/ArenaCrawlerMove-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | The two agents are competing with each other on reaching the target first. | The agents learn basic motion skills of moving towards the target. Also, note that the blue agent learns to stretch one of its arms when it gets close to the target. |
| ArenaCrawlerRush-2T1P-v1-Continuous | <img src="./images/ArenaCrawlerRush-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | The two agents are competing with each other on reaching the target first. | Server: W2n running. |
| ArenaCrawlerPush-2T1P-v1-Continuous | <img src="./images/ArenaCrawlerPush-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | | The agents learn that the most efficient way to make the box reach the target point point is actually not by pushing it, but by kicking it. |
| ArenaWalkerMove-2T1P-v1-Continuous | <img src="./images/ArenaWalkerMove-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | | As shown. |
| Crossroads-2T1P-v1-Continuous | <img src="./images/Crossroads-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | | The agents simply learn to rush towards the goal. |
| ArenaCrawlerPush-2T2P-v1-Continuous | <img src="./images/ArenaCrawlerPush-2T2P-v1-Continuous.gif" align="middle" width="2000"/> | | The agents in the same team learn to push together at the same time, so that the box is pushed forward most efficiently.|
| Crossroads-2T2P-v1-Continuous | <img src="./images/Crossroads-2T2P-v1-Continuous.gif" align="middle" width="2000"/> | | The agents in the same team learn that agent with ID of 0 should go first, so that the agents in the same team would not produce trafic jam with each other. |
| FighterNoTurn-2T1P-v1-Discrete | <img src="./images/FighterNoTurn-2T1P-v1-Discrete.png" align="middle" width="2000"/> | | As shown. |
| FighterFull-2T1P-v1-Discrete | <img src="./images/FighterFull-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | As shown. |
| Soccer-2T1P-v1-Discrete | <img src="./images/Soccer-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | As shown |
| Soccer-2T2P-v1-Discrete | <img src="./images/Soccer-2T2P-v1-Discrete.gif" align="middle" width="2000"/> | | As shown. |
| BlowBlow-2T1P-v1-Discrete | <img src="./images/BlowBlow-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | As shown. |
| BlowBlow-2T2P-v1-Discrete | <img src="./images/BlowBlow-2T2P-v1-Discrete.gif" align="middle" width="2000"/> | | Not Working well, I think there should be some technique needed for credic assignment. |
| Boomer-2T1P-v1-Discrete | <img src="./images/Boomer-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: W5n running |
| Gunner-2T1P-v1-Discrete | <img src="./images/Gunner-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: W4n done |
| Maze2x2Gunner-2T1P-v1-Discrete | <img src="./images/Maze2x2Gunner-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: W4n done |
| Maze3x3Gunner-2T1P-v1-Discrete | <img src="./images/Maze3x3Gunner-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: W4n done |
| Maze3x3Gunner-PenalizeTie-2T1P-v1-Discrete | <img src="./images/Maze3x3Gunner-PenalizeTie-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: W4n running |
| Barrier4x4Gunner-2T1P-v1-Discrete | <img src="./images/Barrier4x4Gunner-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: W4n waiting |
| RunToGoal-2T1P-v1-Continuous | <img src="./images/RunToGoal-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | | Learn to avoid. |
| Sumo-2T1P-v1-Continuous | <img src="./images/Sumo-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | | Server: xx running. |
| Tennis-2T1P-v1-Discrete | <img src="./images/Tennis-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: Wx0 running |
| Tank-FP-2T1P-v1-Discrete | <img src="./images/Tank-FP-2T1P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: Wx1 running |
| YouShallNotPass-Dense-2T1P-v1-Continuous | <img src="./images/YouShallNotPass-Dense-2T1P-v1-Continuous.gif" align="middle" width="2000"/> | | Server: Wx0 running |
| BlowBlow-Dense-2T2P-v1-Discrete | <img src="./images/BlowBlow-Dense-2T2P-v1-Discrete.gif" align="middle" width="2000"/> | | Server: H4n running | -->

## Common Problems

#### Game threads still running

Sometimes, the game threads do not exit properly after you kill the python thread.
Run following command to print a banch of kill commmands.
Then run the printed commands to kill all the game threads.
```
ps aux | grep -ie Linux.x86_64 | awk '{print "kill -9 " $2}'
```

## Issues

For now, we do not support using different spaces across agents
(i.e., all agents have to share the same brain in Arena-BuildingToolkit)
This is because we want to consider the transfer/sharing weight between agents.
If you do have completely different agents in game, one harmless work around is
to use the same brain, but define different meaning of the action in Arena-BuildingToolkit

<!-- #### Copy models

You may find it is useful to copy models from a remote server to your desktop, so that you can see training visualization of the game.
For example,

* The experiment you want to copy is: ```~/ray_results/results/__en-Tennis-2T1P-v1-Discrete__ot-visual__nfs-4__rb-True__no-True__bn-True__nf-False__nk-False__ncc-False__gn-False__ti-ppo__pn-1__rpap-OpenAIFive__pad-True__a-vc```
* The most recent agent id is: ```P0_agent_FRecent```
* You are copying from a remote server: ```-P 30007 yuhangsong@fbafc1ae575e5123.natapp.cc```

You can run following commands to copy necessary checkpoints:

```
# Wx0
mkdir -p ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent
scp -r -P 33007 yuhangsong@ca56526248261483.natapp.cc:~/ray_results/results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent/p_0-i_3388 ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent

# Wx1
mkdir -p ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent
scp -r -P 30007 yuhangsong@fbafc1ae575e5123.natapp.cc:~/ray_results/results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent\{P0_agent_FRecent.pt,eval,checkpoints_reward_record.npy,P0_update_i.npy,event\*\} ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent

# W4n
mkdir -p ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent
scp -r -P 7334 yuhangsong@s1.natapp.cc:~/ray_results/results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent\{P0_agent_FRecent.pt,eval,checkpoints_reward_record.npy,P0_update_i.npy,event\*\} ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent

# W2n
mkdir -p ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent
scp -r -P 7330 yuhangsong@s1.natapp.cc:~/ray_results/results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent\{P0_agent_FRecent.pt,eval,checkpoints_reward_record.npy,P0_update_i.npy,event\*\} ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent

# W5n
mkdir -p ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent
scp -r -P 7333 yuhangsong@s1.natapp.cc:~/ray_results/results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent\{P0_agent_FRecent.pt,eval,checkpoints_reward_record.npy,P0_update_i.npy,event\*\} ~/ray_results/Arena-Benchmark-e=Arena-Tennis-Sparse-2T1P-Discrete-c-nlp=1-c-slp=()-c-aco=()-c-ec-s=(vector)-c-ec-mao=(own)-c-ec-isa=True/PPO_Arena-Tennis-Sparse-2T1P-Discrete_0_iterations_per_reload=1,playing_policy_load_recent_prob=0.8,size_population=1_2020-01-18_08-38-002wadvvwz/learning_agent
``` -->

## Citation

If you use Arena to conduct research, we ask that you cite the following paper as a reference:
```
@article{song2019arena,
  title={Arena: A General Evaluation Platform and Building Toolkit for Multi-Agent Intelligence},
  author={Song, Yuhang and Wang, Jianyi and Lukasiewicz, Thomas and Xu, Zhenghua and Xu, Mai and Ding, Zihan and Wu, Lianlong},
  journal={arXiv preprint arXiv:1905.08085},
  year={2019}
}
```
as well as the engine behind Arena, without which the platform would be impossible to create
```
@article{juliani2018unity,
  title={Unity: A general platform for intelligent agents},
  author={Juliani, Arthur and Berges, Vincent-Pierre and Vckay, Esh and Gao, Yuan and Henry, Hunter and Mattar, Marwan and Lange, Danny},
  journal={arXiv preprint arXiv:1809.02627},
  year={2018}
}
```

## License

[Apache License 2.0](LICENSE)

## Acknowledgement

We give special thanks to the [Whiteson Research Lab](http://whirl.cs.ox.ac.uk/) and [Unity ML-Agents Team](https://unity3d.com/machine-learning/), with which the discussion shaped the vision of the project a lot.
