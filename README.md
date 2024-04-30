# LogiCity

Main Developer: Bowen Li, bowenli2@cs.cmu.edu

<img src="imgs/81.png" alt="81" style="zoom:30%;" />

## Intro

Imagine how you learn to drive a car or play board games, you are usually informed of the rules first. Then you polish and refine your skills for better decisions from experiences. In daily life, people rely on **both explicit reasoning on symbolic rules and implicit learning from experiences** for decision-making. To make models more intelligent and safer, developing a hybrid system that combines learning with reasoning is required. LogiCity environment aims to provide a fully controllable while challenging test bed for such systems, where both reasoning and learning are required.

To be more specific, LogiCity is a simulated 2D urban environment, where all the agents navigate themselves to their goals according to some logical rules, e.g., pedestrians start from one house to one office, cars need to stop if it is close to an agent, ambulances never stops. In other words, the agents need to do both spatial reasoning (planning the least-cost global trajectory) and logical reasoning (how should they behave according to their near environments to avoid obeying rules). We hope such an environment has the following features: **(1) Fully controllable**: we can define any rules easily in first-order logic to control the deterministic behaviors of the agents. **(2) Concept-driven**: we will define some interesting concepts for the agents, which will matter in their rule-based decision-making. **(3) Modular**: The agents can incorporate different global planners or logical planners for their behavior. The rules are written outside the environments as a module so they can be easily modified and changed without rewriting the whole system. **(4) Noise incorporation**: the rules may not be 100% hold, we can create a controllable probability distribution for the rules and agents' actions.

In the end, we hope LogiCity can be used for evaluating cross-community downstream tasks that require both reasoning and learning, including reasoning-based navigation (Robotics), reasoning-based perception (CV), and symbolic grounding (ML).

#### Notification:

This research project and code repo are **ongoing**, please **DO NOT** share with anyone without permission from Bowen.

## Installation

- From scratch

  ```shell
  conda create -n logicity python=3.9
  conda activate logicity
  git clone https://github.com/Jaraxxus-Me/LogiCity.git
  # requirements for logicity
  # torch
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  # others
  pip install -r requirements.txt
  # pyastar, in the LogiCity folder
  mkdir src
  cd src
  git clone https://github.com/Jaraxxus-Me/pyastar2d.git
  cd pyastar2d
  pip install -e .
  # install logicity-lib
  cd ..
  cd ..
  pip install -v -e .
  ```
- Using docker

  ```shell
  docker pull bowenli1024/logicity:v3
  docker run bowenli1024/logicity:v3
  pip install -r requirements.txt
  ```

## LogiCity Simulation

### Running

Only running the simulation for data collection, the cached data will be saved to a `.pkl` file.

```shell
# easy mode
bash scripts/sim/run_sim_easy.sh
# expert mode
bash scripts/sim/run_sim_expert.sh
```

You may make any modifications in the agent configuration.

Some important arguments:

`--vis`: if True, the simulation will save the plain color map and display the agents at each time step, by default, the images will be saved in `./vis`

`--debug`: if True, the start and goal point of each agent will be pre-defined in `./utils/sample.py`, function `sample_determine_start_goal()`

`--max-steps`: Maxium steps of the sim.

`--log_dir`: Directory to save the cached sim.

### Visualization

- Plain color
  The plain color images can be saved by setting `args.vis=True`, then you may use this tool to get a .gif file:
  ```python3
  python3 tools/img2video.py
  ```
- Render some carton-style city / UAV field of view
  ```python3
  # get the carton-style images
  python3 pkl2city.py # you can also have more surprising vis by the pkl2city_uav.py
  # make a video
  python3 tools/img2video.py # change some file name if necessary
  ```

## LogiCity Tasks

### Safe Path Following (SPF)
LogiCity now supports logic based navigation task: the controlled agent is a car, it has 4 action spaces, "Slow" "Fast" "Normal" and "Stop". We require a policy to navigate the ego agent to its goal fast and safe. The are 4 modes:

Easy: Only stop is constrained, few predicates matter (`config/rules/ontology_easy.yaml`)

Medium: Only stop is constrained, more predicates matter (`config/rules/ontology_medium.yaml`)

Hard:  Only stop is constrained, more predicates matter (`config/rules/ontology_full.yaml`)

Expert: Stop, Slow, and Fast are all constrained, more predicates matter (`config/rules/ontology_full.yaml`)

To run an RL task (Using DQN), the other available algorithms Configures are in `config/tasks/Nav/${MODE}/algo`.

```
bash scripts/rl/train_dqn.sh
```

The metrics for this taks are:
- Traj Succ: If the agent gets to goal within 2x oracle steps without violating any rules
- Decision Succ: Count only the traj w/ rule constraints
- Reward: Action Cost * weight + Rule Violation

### Visual Action Prediction
To render the image data with labels:
```
# Switch to vis branch
git checkout vis

# get the vis dataset for scratch, easy
bash scripts/vis/easy/create_vis_dataset_easy_fixed.sh

# train the model with vis dataset
# easy_200_fixed, modular nlm, e2e gnn and e2e nlm
scripts/vis/easy/train_all.sh
```

## Branches

Let Bowen know if you opened a new branch.

- `master`
  is the main and protected branch, it now supports two kind of rule-based simulation. please make sure you pull a request before modifying anything on this branch.
