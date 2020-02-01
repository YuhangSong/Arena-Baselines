<img src="./images/Cover-Landscape-Visual.png" align="middle" width="1000"/>

# Arena-Baselines

<img align="left" width="100" height="100" src="./images/Logo.png">

[Arena](https://sites.google.com/view/arena-unity/) is a general evaluation platform and building toolkit for single/multi-agent intelligence.
As a part of [Arena](https://sites.google.com/view/arena-unity/) project, this repository is to provide implementation of state-of-the-art deep single/multi-agent reinforcement learning baselines.
Arena-Baselines is based on [ray](https://github.com/ray-project/ray).
Other resources can be found in [Arena Home](https://sites.google.com/view/arena-unity/).
If you use this repository to conduct research, we kindly ask that you [cite the papers](#citation) as references.

## Features

Arena-Baselines is built upon [ray](https://github.com/ray-project/ray), benefiting from lots of handy features from it as well as bringing some further advantages.

<img src="./images/arena-ray.png" align="middle" width="900"/>

### Baselines

10+ most popular deep single/multi-agent reinforcement learning baselines, including: independent learners, self-play, variable-shared policies (share arbitrary variables/layers), centralized critic via experiences sharing, centralized critic via observations sharing, Q-Mix, arbitrary grouping agents, centralized critic with counterfactual baselines, population-based training, and etc.

| **Baselines** | **Supported**(:heavy_check_mark:) / **In-progress**(:heavy_minus_sign:) |
| - | -  |
| Independent Learners | :heavy_check_mark: |
| Self-play | :heavy_check_mark: |
| Population-based Training | :heavy_check_mark: |
| Share Weights: between Arbitrary Agents / Teams | :heavy_check_mark: |
| Sharing observations: own, team_absolute, team_relative, all_absolute, all_relative | :heavy_check_mark: |
| Multiple sensors: first-person visual, third-person visual, vector (lidar), or any combinations | :heavy_check_mark: |
| Use separated observations for actor and critic (such as centralized critic and decentralized actors). | :heavy_check_mark: |

### Games

30+ games built by [Arena-BuildingToolkit](https://github.com/YuhangSong/Arena-BuildingToolkit), including Tennis, Tank, BlowBlow, FallFlat, and etc.
There are also multiple variants of each game with different team setups, social structures, actions spaces, observation spaces (first-person visual, third-person visual, vector or mixed), domain randomizations, and etc.
More documentation of games can be found in [Arena-Benchmark](https://github.com/YuhangSong/Arena-Benchmark)

### Utilities

Many research-oriented utilities, such as:

| **Utilities** | **Supported**(:heavy_check_mark:) / **In-progress**(:heavy_minus_sign:) |
| - | - |
| Tune (grid search parameters, schedule trails, yaml config and etc.) | :heavy_check_mark: |
| Interactively load and evaluate trained policies/populations | :heavy_check_mark: |
| Visualize performance in population | :heavy_check_mark: |

## Status: Release

We are currently open to any suggestions or pull requests from the community to make Arena better.
To contribute to the project, [joint us in  Slack](https://join.slack.com/t/arena-ml/shared_invite/enQtNjc1NDE1MTY0MjU3LWMxMjZiMTYyNTE3OWIzM2QxZjU5YmI1NTM2YzYzZDZlNjY0NzllMDFlMjA3MGZiN2QxODA1NTJhZDkzOTI3Nzg), check [To Do list in Trello](https://trello.com/b/zDiljShz) as well as check [issues](https://github.com/YuhangSong/Arena-Baselines/issues) opened in the repo.

## Table of Contents

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Arena-Baselines](#arena-baselines)
	- [Features](#features)
		- [Baselines](#baselines)
		- [Games](#games)
		- [Utilities](#utilities)
	- [Status: Release](#status-release)
	- [Table of Contents](#table-of-contents)
	- [Get Started](#get-started)
		- [Dependencies](#dependencies)
	- [Usage](#usage)
		- [Run tests](#run-tests)
		- [Reproduce Arena-Benchmark](#reproduce-arena-benchmark)
		- [Visualization](#visualization)
		- [Evaluate and Visualize Evaluation](#evaluate-and-visualize-evaluation)
		- [Run in Dummy Mode for Debugging](#run-in-dummy-mode-for-debugging)
		- [Resume Training](#resume-training)
		- [Register New Games](#register-new-games)
	- [Configs](#configs)
	- [Common Problems](#common-problems)
			- [Game threads still running](#game-threads-still-running)
	- [Citation](#citation)
	- [License](#license)
	- [Acknowledgement](#acknowledgement)

<!-- /TOC -->

## Get Started

### Dependencies

To install above dependencies, run: (we are using specific versions of the dependencies)
```bash
/* # Set conda source. Only for users behind the Great Wall of China, no need for other users */
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
/* # If you accidentally did above, type: vim ~/.condarc, and reset conda source by removing the first two lines. Finally, run: conda update --all */

/* # Set pip source. Only for users behind the Great Wall of China, no need for other users. */
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
/* # If you accidentally did above, reset pip source with: pip config set global.index-url https://pypi.org/simple */

/* # Create a virtual environment */
conda remove --name Arena-Baselines --all -y
conda create -n Arena-Baselines python=3.6.9 -y
source activate Arena-Baselines

/* # Create dir */
mkdir Arena
cd Arena

/* # ML-Agents, only compatible with this checkpoint */
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

/* # clone code */
git clone https://github.com/YuhangSong/Arena-Baselines.git
cd Arena-Baselines

/* # PyTorch */
pip install --upgrade torch torchvision

/* # TensorFlow GPU */
pip install tensorflow-gpu==1.14
/* # Or TensorFlow CPU */
/* # pip install tensorflow==1.14 */

/* # Ray and RLlib */
pip install ray[rllib]
pip install ray[debug]
pip install ray==0.7.4

/* # Other requirements */
pip install -r requirements.txt

/* # ffmpeg is needed for monitor */
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

### Visualization

The code log multiple curves (as well as figures and other formats of data) to help analysis the training process, run:
```
source activate Arena-Baselines && tensorboard --logdir=~/Arena-Benchmark --port=9999
```
and visit ```http://localhost:9999``` for visualization with tensorboard.

If your port is blocked, use [natapp](https://natapp.cn/) to forward a port:
```
./natapp --authtoken 237e94b5d173a7c3
```
Above token is a free port used by myself, please apply a free port (or buy a faster one) yourself on [natapp](https://natapp.cn/) so that you can enjoy the best experience.

You should see the ```episode_len_mean``` goes up from 20 (as shown in the following), which means you installation works fine.

<img src="./images/Arena-Benchmark.png" align="middle" width="1000"/>

Our established benchmark can be found in [Arena-Benchmark](https://github.com/YuhangSong/Arena-Benchmark).

### Evaluate and Visualize Evaluation

Append ```--eval``` argument.
In evaluation mode, the programe will detect the logdirs you have locally on your computer as well as detect the populations and iterations you have in each logdir.
Then a series of questions will be promoted for you to make a selection of which checkpoint(s) you want to load for each agent.
After this, you will have selected a list of ```checkpoint_paths``` for each agent, then the programe will run each of the possible combinations and record the results in a single matrix called ```result_matrix```.

```result_matrix``` is nested list with the shape of:
```
(
    len(checkpoint_paths[policy_ids[0]]),
    len(checkpoint_paths[policy_ids[1]]),
    ...,
    len(policy_ids)
)
```
Example: value at ```result_matrix[1,3,0]``` means the episode_rewards_mean of ```policy_0```
when load ```policy_0``` with ```checkpoint_paths[policy_ids[0]][1]```
and load ```policy_1``` with ```checkpoint_paths[policy_ids[1]][3]```.

Then, we will try to visualize ```result_matrix```, see ```vis_result_matrix``` in ```./arena/vis.py```.

You can, of course, compare your agents against our established [Arena-Benchmark](https://github.com/YuhangSong/Arena-Benchmark).

### Run in Dummy Mode for Debugging

Append ```--dummy``` argument to the end of the command, which will try to run the experiments with minimal request of resources. This is for the convenience of debugging.

### Resume Training

To resume/restore a training, append ```--resume``` to the above command in [Reproduce Arena-Benchmark](##reproduce-arena-benchmark)

### Register New Games

To register a new game, append the ID of the game to ```env_ids``` in ```./arena/__init__.py```.
Note that the ID of a game is slightly different from the file name of a game, see [Common Naming Rules](https://github.com/YuhangSong/Arena-Benchmark/blob/master/README.md#common-naming-rules).

## Configs

Function ```create_parser``` in ```./train.py``` gives the detailed description of the configs.
Read them to understand the configs.
Note that we do not recommend passing configs via argparse, instead, use yaml file to config you experiment, as what has been done in [run tests](##run-tests).
In this way, you can make sure your experiments are reproducable.

## Common Problems

#### Game threads still running

Sometimes, the game threads do not exit properly after you kill the python thread.
Run following command to print a banch of kill commmands.
Then run the printed commands to kill all the game threads.
```
ps aux | grep -ie Linux.x86_64 | awk '{print "kill -9 " $2}'
```

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
