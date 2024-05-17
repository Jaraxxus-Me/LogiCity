source /opt/conda/etc/profile.d/conda.sh
conda activate base
# gnn
for r in 0.01 0.05 0.1 0.2 0.5 0.8 1.0
do
# random e2e
python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/hard_200_random_e2e_tl.yaml --exp transfer_e2e_nlm_random_${r}_1 --data_rate $r --seed 1 >> log_vis/transfer_e2e_nlm_random1_${r}.log
# random modular
python3 tools/train_vis_input_nlm.py --config config/tasks/Vis/ResNetNLM/hard_200_random_modular_tl.yaml --modular --exp transfer_modular_nlm_random_${r}_1 --data_rate $r --seed 1 >> log_vis/transfer_modular_nlm_random1_${r}.log
done