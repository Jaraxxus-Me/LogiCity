source /opt/conda/etc/profile.d/conda.sh
conda activate base


python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/veryeasy_200_fixed_e2e.yaml --exp veryeasy_200_fixed_e2e_nlm >> log_vis/veryeasy_200_fixed_e2e_nlm.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/veryeasy_200_fixed_modular.yaml --modular --exp veryeasy_200_fixed_modular_nlm >> log_vis/veryeasy_200_fixed_modular_nlm.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/hard_200_fixed_e2e.yaml --exp hard_200_fixed_e2e_nlm >> log_vis/hard_200_fixed_e2e_nlm.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/hard_200_fixed_modular.yaml --modular --exp hard_200_fixed_modular_nlm >> log_vis/hard_200_fixed_modular_nlm.log