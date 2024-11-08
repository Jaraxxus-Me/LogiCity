# LogiCity

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
```
git checkout master
pip install -v -e .
```

## Visual Action Prediction (VAP), Tab.3, 4, LLM experiments.

In the Visual Action Prediction (VAP) task: the algorithm is required to predict actions for all the agents in an RGB Image (Or language discription).
The code and instuctions for VAP is in `vis` branch:
```
git checkout vis
pip install -v -e .
```

### Dataset
Download the train/val/test datasets [here](https://drive.google.com/file/d/1rgBnPLUQOT6d4WQi888Zhn8VAL3ifbqq/view?usp=sharing)
The folder structure should be like:

```plaintext
LogiCity/
├── vis_dataset/
│   ├── hard_fixed_final/
│   │   ├── train
│   │   │   ├── world0_agent14_imgs
│   │   │   │   ├── step_0001.png
│   │   │   │   ├── step_0002.png
│   │   │   │   └── ...
│   │   │   ├── world1_agent14_imgs
│   │   │   ├── ...
│   │   │   └── test_hard_fixed_final.pkl
│   │   ├── val
│   │   └── test
│   ├── hard_random_final/
│   │   ├── train
│   │   ├── val
│   │   └── test
│   ├── very_easy_random_final/
│   └── very_easy_fixed_final/
├── logicity/
├── config/
└── ...
```

### Pre-trained Models & Test
All of the models displayed in Table 3 can be downloaded [here](https://drive.google.com/file/d/1IeB0DnglkjCMH1n3q_10D1Q94eR7jLG7/view?usp=sharing).
Structure them into:
```plaintext
LogiCity/
├── vis_input_weights/
│   ├── easy/
│   │   ├── spf_emp/
│   │   │   ├── easy/
│   │   │   │   ├── veryeasy_200_fixed_e2e_gnn_epoch19_valacc0.7727.pth
│   │   │   │   ├── veryeasy_200_fixed_e2e_gnn_epoch24_valacc0.7755.pth
│   │   │   │   └── ...
│   │   │   └── hard/
├── logicity/
├── config/
└── ...
```

To test them, several example commands could be:
```
# this test e2e GNN in easy mode
python3 tools/test_vis_input_e2e.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml --ckpt vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn_epoch19_valacc0.7727.pth --exp easy_gnn_test_veryeasy_200_fixed_e2e_gnn_epoch19_valacc0.7727
# this test modular GNN in easy mode
python3 tools/test_vis_input_mod.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_modular2.yaml --ckpt vis_input_weights/easy/veryeasy_200_fixed_modular_gnn2_epoch29_valacc0.7989.pth --exp easy_gnn_test_veryeasy_200_fixed_modular_gnn2_epoch29_valacc0.7989
```
Note that all the models are tested using `fixed` configuration in Table 3.

The output will be like:
```
Action 2 is unseen.
Slow: Correct_num: 2183, Total_num: 3042, Acc: 0.7176
Normal: Correct_num: 2867, Total_num: 3978, Acc: 0.7207
Fast: Correct_num: 0, Total_num: 0, Acc: nan
Stop: Correct_num: 7125, Total_num: 7220, Acc: 0.9868
Testing Sample Avg Acc: 0.8550
Action Weighted Acc: 0.7706
```

### Train a New Model
All the configurations for all the models are at `config/tasks/Vis`.
```
# Training NLM models
scripts/vis/easy/train_nlm.sh
# Training GNN models
scripts/vis/easy/train_gnn.sh
```
The checkpoints will be saved in `vis_input_weights`.

### Testing LLMs and Human
1. Convert VAP dataset to [MMLU](https://crfm.stanford.edu/helm/mmlu/latest/) format
  You can skip this part if you are using our default dataset, the converted QA pairs can be downloaded [here](), put `mmlu_logicity` and `mmlu_logicity_human` under folder `vis_dataset`.
  With your customized VAP dataset, run the program (change some variables):
  ```
  tools/vispkl2mmlu.py
  ```
2. Test OpenAI LLMs:
  ```
  tools/test_mmlu_gpt.py
  ```
  Note that we have used the human test and train QA pairs. The Raw results over 3 random seeds for Tab.3 can be downloaded [here]().
3. Get the results
  ```
  tools/pklnpy2results_part.py
  ```
To test a human, we also provided the tools to generate documents with specified concepts:
```
tools/get_human_data.py
```
