source /opt/conda/etc/profile.d/conda.sh
conda activate base
# gnn
for r in 0.01 0.05 0.1 0.2 0.5 0.8 1.0
do
# fixed e2e
python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e_tl.yaml --exp transfer_e2e_gnn_fixed_${r} --data_rate $r >> log_vis/transfer_e2e_gnn_fixed_${r}.log
# fixed modular
python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_fixed_modular2_tl.yaml --modular --exp transfer_modular_gnn_fixed_${r} --data_rate $r >> log_vis/transfer_modular_gnn_fixed_${r}.log
done