source /opt/conda/etc/profile.d/conda.sh
conda activate base
# gnn
for r in 0.01 0.05 0.1 0.2 0.5 0.8 1.0
do
# fixed e2e
python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e_tl2.yaml --exp transfer2_e2e_gnn_fixed_${r} --data_rate $r --seed 1 >> log_vis/transfer2_e2e_gnn_fixed1_${r}.log
# fixed modular
python3 tools/train_vis_input_gnn.py --config config/tasks/Vis/ResNetGNN/hard_200_fixed_modular2_tl2.yaml --modular --exp transfer2_modular_gnn_fixed_${r} --data_rate $r --seed 1 >> log_vis/transfer2_modular_gnn_fixed1_${r}.log
done