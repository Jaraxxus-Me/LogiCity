source /opt/conda/etc/profile.d/conda.sh
conda activate base

# easy fixed gnn e2e
# checkpoints=(
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn/veryeasy_200_fixed_e2e_gnn_epoch28_valacc0.7817.pth"
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn/veryeasy_200_fixed_e2e_gnn_epoch24_valacc0.7755.pth"
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_gnn/veryeasy_200_fixed_e2e_gnn_epoch19_valacc0.7727.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_gnn/veryeasy_200_random_e2e_gnn_epoch29_valacc0.8046.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_gnn/veryeasy_200_random_e2e_gnn_epoch24_valacc0.8028.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_gnn/veryeasy_200_random_e2e_gnn_epoch23_valacc0.8020.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetGNN/veryeasy_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# # easy fixed nlm e2e
# checkpoints=(
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch27_valacc0.6945.pth"
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch29_valacc0.6821.pth"
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch26_valacc0.6811.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch28_valacc0.7992.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch22_valacc0.7944.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch20_valacc0.7908.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetNLM/veryeasy_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_test.log"
# done

# hard fixed gnn e2e
checkpoints=(
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.1/transfer_e2e_gnn_fixed_0.1_epoch14_valacc0.2268.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.1/transfer_e2e_gnn_fixed_0.1_epoch12_valacc0.2245.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.1/transfer_e2e_gnn_fixed_0.1_epoch13_valacc0.2180.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.01/transfer_e2e_gnn_fixed_0.01_epoch0_valacc0.2221.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.01/transfer_e2e_gnn_fixed_0.01_epoch1_valacc0.2153.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.01/transfer_e2e_gnn_fixed_0.01_epoch2_valacc0.2104.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.2/transfer_e2e_gnn_fixed_0.2_epoch8_valacc0.2287.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.2/transfer_e2e_gnn_fixed_0.2_epoch6_valacc0.2230.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.2/transfer_e2e_gnn_fixed_0.2_epoch9_valacc0.2132.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.5/transfer_e2e_gnn_fixed_0.5_epoch2_valacc0.2255.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.5/transfer_e2e_gnn_fixed_0.5_epoch3_valacc0.2140.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.5/transfer_e2e_gnn_fixed_0.5_epoch28_valacc0.2089.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.05/transfer_e2e_gnn_fixed_0.05_epoch0_valacc0.2075.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.05/transfer_e2e_gnn_fixed_0.05_epoch1_valacc0.2020.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.05/transfer_e2e_gnn_fixed_0.05_epoch2_valacc0.2015.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.8/transfer_e2e_gnn_fixed_0.8_epoch20_valacc0.2242.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.8/transfer_e2e_gnn_fixed_0.8_epoch19_valacc0.2194.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_0.8/transfer_e2e_gnn_fixed_0.8_epoch24_valacc0.2151.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_1.0/transfer_e2e_gnn_fixed_1.0_epoch12_valacc0.2301.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_1.0/transfer_e2e_gnn_fixed_1.0_epoch1_valacc0.2126.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_fixed_1.0/transfer_e2e_gnn_fixed_1.0_epoch21_valacc0.2105.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_transfer_test.log"
done

# hard fixed nlm e2e
# checkpoints=(
#     "vis_input_weights/easy/veryeasy_200_fixed_e2e_nlm/veryeasy_200_fixed_e2e_nlm_epoch26_valacc0.6811.pth"
#     "vis_input_weights/easy/veryeasy_200_random_e2e_nlm/veryeasy_200_random_e2e_nlm_epoch22_valacc0.7944.pth"
# )

# # Loop over each checkpoint file
# for ckpt in "${checkpoints[@]}"
# do
#     exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
#     python3 tools/test_vis_input_e2e.py \
#         --config config/tasks/Vis/ResNetNLM/hard_200_fixed_e2e.yaml \
#         --ckpt "$ckpt" \
#         --exp "$exp_name" \
#         >> "log_vis/${exp_name}_transfer_init_test.log"
# done