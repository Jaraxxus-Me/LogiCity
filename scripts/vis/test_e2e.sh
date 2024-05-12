source /opt/conda/etc/profile.d/conda.sh
conda activate base

# easy fixed gnn e2e
# checkpoints=(
#     "vis_input_weights/easy/easy_200_random_e2e_gnn/easy_200_random_e2e_gnn_epoch16_valacc0.4731.pth"
#     "vis_input_weights/easy/easy_200_random_e2e_gnn/easy_200_random_e2e_gnn_epoch24_valacc0.4172.pth"
#     "vis_input_weights/easy/easy_200_random_e2e_gnn/easy_200_random_e2e_gnn_epoch20_valacc0.3834.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# # easy fixed nlm e2e
# checkpoints=(
#     "vis_input_weights/easy/easy_200_random_e2e_nlm/easy_200_random_e2e_nlm_epoch6_valacc0.2880.pth"
#     "vis_input_weights/easy/easy_200_random_e2e_nlm/easy_200_random_e2e_nlm_epoch22_valacc0.2818.pth"
#     "vis_input_weights/easy/easy_200_random_e2e_nlm/easy_200_random_e2e_nlm_epoch29_valacc0.2703.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetNLM/easy_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# hard fixed gnn e2e
checkpoints=(
    "vis_input_weights/hard/hard_200_random_e2e_gnn/hard_200_random_e2e_gnn_epoch17_valacc0.4426.pth"
    "vis_input_weights/hard/hard_200_random_e2e_gnn/hard_200_random_e2e_gnn_epoch24_valacc0.4106.pth"
    "vis_input_weights/hard/hard_200_random_e2e_gnn/hard_200_random_e2e_gnn_epoch26_valacc0.4000.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# hard fixed nlm e2e
checkpoints=(
    "vis_input_weights/hard/hard_200_random_e2e_nlm/hard_200_random_e2e_nlm_epoch7_valacc0.3889.pth"
    "vis_input_weights/hard/hard_200_random_e2e_nlm/hard_200_random_e2e_nlm_epoch9_valacc0.3880.pth"
    "vis_input_weights/hard/hard_200_random_e2e_nlm/hard_200_random_e2e_nlm_epoch18_valacc0.3677.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetNLM/hard_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done