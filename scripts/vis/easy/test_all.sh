source /opt/conda/etc/profile.d/conda.sh
conda activate logicity

python tools/test_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_200_fixed_e2e.yaml \
    --ckpt vis_input_weights/easy/easy_200_fixed_nlm_e2e/easy_200_fixed_nlm_e2e_best.pth \
    >> log_sim/easy_200_fixed_nlm_e2e_test.log

python tools/test_vis_input.py --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml \
    --ckpt vis_input_weights/easy/easy_200_fixed_gnn_e2e/easy_200_fixed_gnn_e2e_best.pth \
    >> log_sim/easy_200_fixed_gnn_e2e_test.log

python tools/test_vis_input.py --config config/tasks/Vis/ResNetNLM/easy_200_fixed_modular.yaml \
    --ckpt vis_input_weights/easy/easy_200_fixed_nlm_modular/easy_200_fixed_nlm_modular_best.pth \
    >> log_sim/easy_200_fixed_nlm_modular_test.log
