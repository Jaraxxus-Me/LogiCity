source /opt/conda/etc/profile.d/conda.sh
conda activate base


python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml --exp easy_200_fixed_e2e_gnn >> log_vis/easy_200_fixed_e2e_gnn.log

python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_modular2.yaml --modular --exp easy_200_fixed_modular_gnn2 >> log_vis/easy_200_fixed_modular_gnn2.log

python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e.yaml --exp hard_200_fixed_e2e_gnn >> log_vis/hard_200_fixed_e2e_gnn.log

python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_fixed_modular2.yaml --modular --exp hard_200_fixed_modular_gnn2 >> log_vis/hard_200_fixed_modular_gnn2.log