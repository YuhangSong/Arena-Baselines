# alley
Baselines for MultiAgent Systems

## Introduction

## Installation

Follow:

```bash
# For users behind the Great Wall of China
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Create a virtual environment
conda create -n Arena python=3.6.7 -y
source activate Arena

# PyTorch
pip install --upgrade torch torchvision

# TensorFlow
pip install --upgrade tensorflow

# Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..

# ML-Agents
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

# Clone code
mkdir Arena
cd Arena
git clone https://github.com/abi-aryan/alley.git
cd alley

# Other requirements
pip install -r requirements.txt
```

If you run into following situations,

* you are running on a remote server without GUI (X-Server).
* your machine have a X-Server but it does not belongs (started by) your account, so you cannot access it.
* if none of above is your situation, i.e., you are running things on your onw desktop, skip this section.

follow guidelines [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) to setup a virtual display.

## Usage

Crate TMUX session (if the machine is a server you connect via SSH) and enter virtual environment
```
tmux new-session -s Arena
source activate Arena
```

### Continuous Action Space

#### Baseline: Self-Play

Games:
* ArenaCrawler-Example-v2-Continuous
* ArenaCrawlerMove-2T1P-v1-Continuous
* ArenaCrawlerRush-2T1P-v1-Continuous
* ArenaCrawlerPush-2T1P-v1-Continuous
* ArenaWalkerMove-2T1P-v1-Continuous
* Crossroads-2T1P-v1-Continuous
* Crossroads-2T2P-v1-Continuous
* ArenaCrawlerPush-2T2P-v1-Continuous
* RunToGoal-2T1P-v1-Continuous
* Sumo-2T1P-v1-Continuous
* YouShallNotPass-Dense-2T1P-v1-Continuous

Commands, replace GAME_NAME with above games:
```
```

### Discrete Action Space

#### Baseline: Self-Play

Games:
* Crossroads-2T1P-v1-Discrete
* FighterNoTurn-2T1P-v1-Discrete
* FighterFull-2T1P-v1-Discrete
* Soccer-2T1P-v1-Discrete
* BlowBlow-2T1P-v1-Discrete
* Boomer-2T1P-v1-Discrete
* Gunner-2T1P-v1-Discrete
* Maze3x3Gunner-PenalizeTie-2T1P-v1-Discrete
* Maze3x3Gunner-PenalizeTie-2T1P-v2-Discrete
* Barrier4x4Gunner-2T1P-v1-Discrete
* Soccer-2T2P-v1-Discrete
* BlowBlow-2T2P-v1-Discrete
* BlowBlow-Dense-2T2P-v1-Discrete
* Tennis-2T1P-v1-Discrete
* Tank-FP-2T1P-v1-Discrete
* BlowBlow-Dense-2T1P-v1-Discrete

We also support traditional Atari games, such as:
* PongNoFrameskip-v4
* BreakoutNoFrameskip-v4

Commands, replace GAME_NAME with above games:

```
```
