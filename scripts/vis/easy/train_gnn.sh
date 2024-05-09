source /opt/conda/etc/profile.d/conda.sh
conda activate logicity


python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml >> log_vis/easy_200_fixed_e2e_gnn.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_modular.yaml --modular >> log_vis/easy_200_fixed_modular_gnn.log