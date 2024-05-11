source /opt/conda/etc/profile.d/conda.sh
conda activate base
# easy random
# python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/easy_200_random_e2e.yaml --exp easy_200_random_e2e_gnn >> log_vis/easy_200_random_e2e_gnn.log

# python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/easy_200_random_modular2.yaml --modular --exp easy_200_random_modular_gnn2 >> log_vis/easy_200_random_modular_gnn2.log

# python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/easy_200_random_e2e.yaml --exp easy_200_random_e2e_nlm >> log_vis/easy_200_random_e2e_nlm.log

# python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/easy_200_random_modular.yaml --modular --exp easy_200_random_modular_nlm >> log_vis/easy_200_random_modular_nlm.log
# # hard random
# python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_random_e2e.yaml --exp hard_200_random_e2e_gnn >> log_vis/hard_200_random_e2e_gnn.log

python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_random_modular2.yaml --modular --exp hard_200_random_modular_gnn2 >> log_vis/hard_200_random_modular_gnn2.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/hard_200_random_e2e.yaml --exp hard_200_random_e2e_nlm >> log_vis/hard_200_random_e2e_nlm.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/hard_200_random_modular.yaml --modular --exp hard_200_random_modular_nlm >> log_vis/hard_200_random_modular_nlm.log

# very easy random
python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/veryeasy_200_random_e2e.yaml --exp veryeasy_200_random_e2e_gnn >> log_vis/veryeasy_200_random_e2e_gnn.log

python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/veryeasy_200_random_modular2.yaml --modular --exp veryeasy_200_random_modular_gnn2 >> log_vis/veryeasy_200_random_modular_gnn2.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/veryeasy_200_random_e2e.yaml --exp veryeasy_200_random_e2e_nlm >> log_vis/veryeasy_200_random_e2e_nlm.log

python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/veryeasy_200_random_modular.yaml --modular --exp veryeasy_200_random_modular_nlm >> log_vis/veryeasy_200_random_modular_nlm.log