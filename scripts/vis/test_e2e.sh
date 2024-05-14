source /opt/conda/etc/profile.d/conda.sh
conda activate base

# easy fixed gnn e2e
checkpoints=(
    "vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn/veryeasy_200_fixed_e2e_gnn_epoch28_valacc0.7817.pth"
    "vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn/veryeasy_200_fixed_e2e_gnn_epoch24_valacc0.7755.pth"
    "vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn/veryeasy_200_fixed_e2e_gnn_epoch19_valacc0.7727.pth"
    "vis_input_weights/easy/veryeasy_200_random_e2e_gnn/veryeasy_200_random_e2e_gnn_epoch29_valacc0.8046.pth"
    "vis_input_weights/easy/veryeasy_200_random_e2e_gnn/veryeasy_200_random_e2e_gnn_epoch24_valacc0.8028.pth"
    "vis_input_weights/easy/veryeasy_200_random_e2e_gnn/veryeasy_200_random_e2e_gnn_epoch23_valacc0.8020.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetGNN/veryeasy_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# easy fixed nlm e2e
checkpoints=(
    "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch27_valacc0.6945.pth"
    "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch29_valacc0.6821.pth"
    "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch26_valacc0.6811.pth"
    "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch28_valacc0.7992.pth"
    "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch22_valacc0.7944.pth"
    "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch20_valacc0.7908.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetNLM/veryeasy_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# hard fixed gnn e2e
# checkpoints=(
#     "vis_input_weights/hard/hard_200_random_e2e_gnn/hard_200_random_e2e_gnn_epoch17_valacc0.4426.pth"
#     "vis_input_weights/hard/hard_200_random_e2e_gnn/hard_200_random_e2e_gnn_epoch24_valacc0.4106.pth"
#     "vis_input_weights/hard/hard_200_random_e2e_gnn/hard_200_random_e2e_gnn_epoch26_valacc0.4000.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# # hard fixed nlm e2e
# checkpoints=(
#     "vis_input_weights/hard/hard_200_random_e2e_nlm/hard_200_random_e2e_nlm_epoch7_valacc0.3889.pth"
#     "vis_input_weights/hard/hard_200_random_e2e_nlm/hard_200_random_e2e_nlm_epoch9_valacc0.3880.pth"
#     "vis_input_weights/hard/hard_200_random_e2e_nlm/hard_200_random_e2e_nlm_epoch18_valacc0.3677.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetNLM/hard_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done