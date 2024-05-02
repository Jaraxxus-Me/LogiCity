source /opt/conda/etc/profile.d/conda.sh
conda activate logicity

# python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_200_fixed_modular.yaml \
#     --exp easy_200_fixed_nlm_modular \
#     --modular >> log_sim/easy_200_fixed_nlm_modular.log

# python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_200_fixed_e2e.yaml \
#     --exp easy_200_fixed_nlm_e2e >> log_sim/easy_200_fixed_nlm_e2e.log

# python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml \
#     --exp easy_200_fixed_gnn_e2e >> log_sim/easy_200_fixed_gnn_e2e.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_modular.yaml \
    --exp easy_100_fixed_demo_nlm_modular \
    --modular >> log_sim/easy_100_fixed_demo_nlm_modular.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_demo_e2e.yaml \
    --exp easy_100_fixed_demo_nlm_e2e >> log_sim/easy_100_fixed_demo_nlm_e2e.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_no_variance_modular.yaml \
    --exp easy_100_fixed_no_variance_nlm_modular \
    --modular >> log_sim/easy_100_fixed_no_variance_nlm_modular.log

python3 tools/train_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_100_fixed_no_variance_e2e.yaml \
    --exp easy_100_fixed_no_variance_nlm_e2e >> log_sim/easy_100_fixed_no_variance_nlm_e2e.log
