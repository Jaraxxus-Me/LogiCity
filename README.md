# LogiCity
Officially published in NeurIPS'24 D&B Track.

<img src="imgs/81.png" alt="81" style="zoom:30%;" />

## Abstract

  Recent years have witnessed the rapid development of Neuro-Symbolic (NeSy) AI systems, which integrate symbolic reasoning into deep neural networks.
  However, most of the existing benchmarks for NeSy AI fail to provide long-horizon reasoning tasks with complex multi-agent interaction.
  Furthermore, they are usually constrained by fixed and simplistic logical rules over limited entities, making them inadequate for capturing real-world complexities.
  To address these crucial gaps, we introduce LogiCity, the first simulator based on customizable first-order logic (FOL) for urban environments with multiple dynamic agents.
  LogiCity models various urban elements, including buildings, cars, and pedestrians, using semantic and spatial concepts, such as $\texttt{IsAmbulance}(\texttt{X})$ and $\texttt{IsClose}(\texttt{X}, \texttt{Y})$. 
  These concepts are used to define FOL rules governing the behavior of multiple dynamic agents. 
  Since the concepts and rules are abstractions, cities with distinct agent compositions can be easily instantiated and simulated. 
  Besides, a key benefit is that LogiCity allows for user-configurable abstractions, which enables customizable simulation complexities about logical reasoning.
  To explore various aspects of NeSy AI, we design long-horizon sequential decision-making and one-step visual reasoning tasks, varying in difficulty and agent behaviors.
  Our extensive evaluation using LogiCity reveals the advantage of NeSy frameworks in abstract reasoning. 
  Moreover, we highlight the significant challenges of handling more complex abstractions in long-horizon multi-agent reasoning scenarios or under high-dimensional, imbalanced data.
  With the flexible design, various features, and newly raised challenges, we believe LogiCity represents a pivotal step for advancing the next generation of NeSy AI.

## Installation

- From scratch

  ```shell
  # requirements for logicity
  # using conda env
  conda create -n logicity python=3.11.5
  conda activate logicity
  # pyastar, in the LogiCity folder
  mkdir src
  cd src
  git clone https://github.com/Jaraxxus-Me/pyastar2d.git
  cd pyastar2d
  # install pyastar
  pip install -e .
  # install logicity-lib
  cd ..
  cd ..
  pip install -v -e .
  ```
- Using docker

  ```shell
  docker pull bowenli1024/logicity:latest
  docker run bowenli1024/logicity:latest
  # inside the docker container
  conda activate logicity
  cd path/to/LogiCity
  pip install -v -e .
  ```

## Simulation

### Running

Running the simulation for santity check, the cached data will be saved to a `.pkl` file.

```shell
mkdir log_sim
# easy mode
# the configuration is config/tasks/sim/easy.yaml, pkl saved to log_sim
bash scripts/sim/run_sim_easy.sh
# expert mode
# the configuration is config/tasks/sim/expert.yaml, pkl saved to log_sim
bash scripts/sim/run_sim_expert.sh
```

### Visualization

- Render some default carton-style city
  ```python3
  # get the carton-style images
  mkdir vis
  python3 tools/pkl2city.py --pkl log_sim/easy_100_0.pkl --output_folder vis # modify to your pkl file
  # make a video
  python3 tools/img2video.py vis demo.gif # change some file name if necessary
  ```

### Customize a City
The configurations (abstractions) of a City is defined (for example, the easy demo) here: `config/tasks/sim/*.yaml`.
```yaml
simulation:
  map_yaml_file: "config/maps/square_5x5.yaml"       # OpenAI Gym environment name
  agent_yaml_file: "config/agents/easy/train.yaml" # Agents in the simulation
  ontology_yaml_file: "config/rules/ontology_easy.yaml" # Ontology of the simulation
  rule_type: "Z3"               # z3 rl will set the rl_agent with fixed number of other entities, return the groundings as obs, and return the rule reward
  rule_yaml_file: "config/rules/sim/easy/easy_rule.yaml"                 # Whether to render the environment
  rl: false
  debug: false
  use_multi: false
  agent_region: 100
```
Things you might want to play with:
- `agent_yaml_file` defines the agent configuration, you can arbitarily define your own configurations.
- `rule_yaml_file` defines the FOL rules of the city. You can customize your own rule, but the naming should follow [z3](https://ericpony.github.io/z3py-tutorial/guide-examples.htm#:~:text=Satisfiability%20and%20Validity).
- `ontology_yaml_file` defines the possible concepts in the city (used by the rules). You can also customize the *grounding* functions specified in the function fields.

## Safe Path Following (SPF, master branch, Tab. 2 in paper)

In the Safe Path Following (SPF) task: the controlled agent is a car, it has 4 action spaces, "Slow" "Fast" "Normal" and "Stop". We require a policy to navigate the ego agent to its goal with minimum trajectory cost.
This is an RL wrapper using the simulation above. We have used [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) coding format.

### Dataset
Download the train/val/test episodes [here](https://drive.google.com/file/d/1ePLVlNH77VV25171yOSgku21tji9ISdG/view?usp=sharing) and unzip it.
The folder structure should be like:

```plaintext
LogiCity/
├── dataset/
│   ├── easy/
│   │   ├── test_100_episodes.pkl
│   │   ├── val_40_episodes.pkl
│   │   └── train_1ktraj.pkl
│   ├── expert/
│   │   ├── test_100_episodes.pkl
│   │   ├── val_40_episodes.pkl
│   │   └── train_1ktraj.pkl
│   └── ...
├── logicity/
├── config/
└── ...
```

### Pre-trained Models & Test
All of the models displayed in Tab. 2 can be downloaded [here](https://drive.google.com/file/d/1gDMu4AlljMR1FeUh5ty1y7sO0KW5CV4d/view?usp=sharing).
Structure them into:
```plaintext
LogiCity/
├── checkpoints/
│   ├── final_models/
│   │   ├── spf_emp/
│   │   │   ├── easy/
│   │   │   │   ├── dqn.zip
│   │   │   │   ├── nlmdqn.zip
│   │   │   │   └── ...
│   │   │   ├── expert/
│   │   │   ├── hard/
│   │   │   └── medium/
├── logicity/
├── config/
└── ...
```

To test them, an example command could be:
```
# this test NLM-DQN in expert mode
python3 main.py --config config/tasks/Nav/expert/algo/nlmdqn_test.yaml --exp nlmdqn_expert_test \
    --checkpoint_path checkpoints/final_models/spf_emp/expert/nlmdqn.zip --use_gym
```

The metrics for this taks are:
- Traj Succ: If the agent gets to goal within 2x oracle steps without violating any rules
- Decision Succ: Count only the traj w/ rule constraints
- Reward: Action Cost * weight + Rule Violation

The output will be at `log_rl/nlmdqn_expert_test.log`.

### Train a New Model
All the configurations for all the models are at `config/tasks/Nav`.
We provide two examples to train models:
```
# Training GNN-Behaviro Cloning Agent in easy mode
python3 main.py --config config/tasks/Nav/easy/algo/gnnbc.yaml --exp gnnbc_easy_train --use_gym
# Training DQN Agent in easy mode, with 2 parallel envs
python3 main.py --config config/tasks/Nav/easy/algo/dqn.yaml --exp gnnbc_easy_train --use_gym
```
Outputs from RL training is like the following:
```shell
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 41.5     |
|    ep_rew_mean      | -10.2    |
|    exploration_rate | 0.998    |
|    success_rate     | 0        |
| time/               |          |
|    episodes         | 4        |
|    fps              | 9        |
|    time_elapsed     | 18       |
|    total_timesteps  | 184      |
----------------------------------
```
The checkpoints will be saved in `checkpoints`. By default, the validation episodes are used and the results are saved also in `checkpoints`.

### Customize you own City and study RL
Configurations for RL training and testing are in this folder: `config/tasks/Nav`.
Similar to the simulation process, you can customize agent compositions, rules, and concepts by changing the fields in `config/tasks/Nav/easy/algo`
using different `.yaml` files.
We also probided a bunch of tools (collecting demonstrations, for example) in `scripts/rl`. You might find them useful.

## Visual Action Prediction (VAP), Tab.3, 4, LLM experiments.

In the Visual Action Prediction (VAP) task: the algorithm is required to predict actions for all the agents in an RGB Image (Or language discription).
The code and instuctions for VAP is in `vis` branch:
```
git checkout vis
pip install -v -e .
```
## Reference
If you used our work in your research, or you find our work useful, please cite us as:
```
@INPROCEEDINGS{Li2023logicity,     
  title={{LogiCity: Advancing Neuro-Symbolic AI with Abstract Urban Simulation}},
  author={Li, Bowen and Li, Zhaoyu and Du, Qiwei and Luo, Jinqi and Wang, Wenshan and Xie, Yaqi and Stepputtis, Simon and Wang, Chen and Sycara, Katia P and Ravikumar, Pradeep Kumar and Gray, Alex and Si, Xujie and Scherer, Sebastian}, 
  booktitle={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)}, 
  year={2024},
  volume={},
  number={}
}
```
