source /opt/conda/etc/profile.d/conda.sh
conda activate base

# easy fixed gnn modular
# checkpoints=(
#     "vis_input_weights/easy/easy_200_random_modular_gnn2/easy_200_random_modular_gnn2_epoch29_valacc0.2030.pth"
#     "vis_input_weights/easy/easy_200_random_modular_gnn2/easy_200_random_modular_gnn2_epoch22_valacc0.1980.pth"
#     "vis_input_weights/easy/easy_200_random_modular_gnn2/easy_200_random_modular_gnn2_epoch18_valacc0.1860.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_mod.py \
#         --config config/tasks/Vis/ResNetGNN/easy_200_fixed_modular2.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# # easy fixed nlm modular
# checkpoints=(
#     "vis_input_weights/easy/easy_200_random_modular_nlm/easy_200_random_modular_nlm_epoch27_valacc0.2302.pth"
#     "vis_input_weights/easy/easy_200_random_modular_nlm/easy_200_random_modular_nlm_epoch29_valacc0.2275.pth"
#     "vis_input_weights/easy/easy_200_random_modular_nlm/easy_200_random_modular_nlm_epoch23_valacc0.2155.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_mod.py \
#         --config config/tasks/Vis/ResNetNLM/easy_200_fixed_modular.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# hard fixed gnn modular
checkpoints=(
    "vis_input_weights/hard/hard_200_random_modular_gnn2/hard_200_random_modular_gnn2_epoch25_valacc0.2954.pth"
    "vis_input_weights/hard/hard_200_random_modular_gnn2/hard_200_random_modular_gnn2_epoch28_valacc0.2944.pth"
    "vis_input_weights/hard/hard_200_random_modular_gnn2/hard_200_random_modular_gnn2_epoch24_valacc0.2930.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_mod.py \
        --config config/tasks/Vis/ResNetGNN/hard_200_fixed_modular2.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# hard fixed nlm modular
checkpoints=(
    "vis_input_weights/hard/hard_200_random_modular_nlm/hard_200_random_modular_nlm_epoch13_valacc0.3001.pth"
    "vis_input_weights/hard/hard_200_random_modular_nlm/hard_200_random_modular_nlm_epoch22_valacc0.2876.pth"
    "vis_input_weights/hard/hard_200_random_modular_nlm/hard_200_random_modular_nlm_epoch14_valacc0.2965.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_mod.py \
        --config config/tasks/Vis/ResNetNLM/hard_200_fixed_modular.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done