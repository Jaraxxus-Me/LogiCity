source /opt/conda/etc/profile.d/conda.sh
conda activate base

# easy fixed gnn e2e
checkpoints=(
    "vis_input_weights/easy/easy_200_fixed_e2e_gnn/easy_200_fixed_e2e_gnn_epoch0_valacc0.3548.pth"
    "vis_input_weights/easy/easy_200_fixed_e2e_gnn/easy_200_fixed_e2e_gnn_epoch21_valacc0.3542.pth"
    "vis_input_weights/easy/easy_200_fixed_e2e_gnn/easy_200_fixed_e2e_gnn_epoch9_valacc0.3389.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetGNN/easy_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# easy fixed nlm e2e
checkpoints=(
    "vis_input_weights/easy/easy_200_fixed_e2e_nlm/easy_200_fixed_e2e_nlm_epoch29_valacc0.3665.pth"
    "vis_input_weights/easy/easy_200_fixed_e2e_nlm/easy_200_fixed_e2e_nlm_epoch0_valacc0.3617.pth"
    "vis_input_weights/easy/easy_200_fixed_e2e_nlm/easy_200_fixed_e2e_nlm_epoch2_valacc0.3338.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py \
        --config config/tasks/Vis/ResNetNLM/easy_200_fixed_e2e.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# hard fixed gnn e2e
checkpoints=(
    "vis_input_weights/hard/hard_200_fixed_e2e_gnn/hard_200_fixed_e2e_gnn_epoch15_valacc0.3045.pth"
    "vis_input_weights/hard/hard_200_fixed_e2e_gnn/hard_200_fixed_e2e_gnn_epoch0_valacc0.2532.pth"
    "vis_input_weights/hard/hard_200_fixed_e2e_gnn/hard_200_fixed_e2e_gnn_epoch26_valacc0.2484.pth"
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
    "vis_input_weights/hard/hard_200_fixed_e2e_nlm/hard_200_fixed_e2e_nlm_epoch28_valacc0.3030.pth"
    "vis_input_weights/hard/hard_200_fixed_e2e_nlm/hard_200_fixed_e2e_nlm_epoch15_valacc0.2917.pth"
    "vis_input_weights/hard/hard_200_fixed_e2e_nlm/hard_200_fixed_e2e_nlm_epoch12_valacc0.2860.pth"
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