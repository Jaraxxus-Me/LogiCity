source /opt/conda/etc/profile.d/conda.sh
conda activate base

# easy fixed gnn modular
checkpoints=(
    "vis_input_weights/easy/easy_200_fixed_modular_gnn2/easy_200_fixed_modular_gnn2_epoch28_valacc0.2074.pth"
    "vis_input_weights/easy/easy_200_fixed_modular_gnn2/easy_200_fixed_modular_gnn2_epoch24_valacc0.2181.pth"
    "vis_input_weights/easy/easy_200_fixed_modular_gnn2/easy_200_fixed_modular_gnn2_epoch23_valacc0.2000.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_mod.py \
        --config config/tasks/Vis/ResNetGNN/easy_200_fixed_modular2.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# easy fixed nlm modular
checkpoints=(
    "vis_input_weights/easy/easy_200_fixed_modular_nlm/easy_200_fixed_modular_nlm_epoch20_valacc0.2180.pth"
    "vis_input_weights/easy/easy_200_fixed_modular_nlm/easy_200_fixed_modular_nlm_epoch9_valacc0.2131.pth"
    "vis_input_weights/easy/easy_200_fixed_modular_nlm/easy_200_fixed_modular_nlm_epoch14_valacc0.2242.pth"
)

# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_mod.py \
        --config config/tasks/Vis/ResNetNLM/easy_200_fixed_modular.yaml \
        --ckpt "$ckpt" \
        --exp "$exp_name" \
        >> "log_vis/${exp_name}_test.log"
done

# hard fixed gnn modular
checkpoints=(
    "vis_input_weights/hard/hard_200_fixed_modular_gnn2/hard_200_fixed_modular_gnn2_epoch27_valacc0.2838.pth"
    "vis_input_weights/hard/hard_200_fixed_modular_gnn2/hard_200_fixed_modular_gnn2_epoch29_valacc0.2762.pth"
    "vis_input_weights/hard/hard_200_fixed_modular_gnn2/hard_200_fixed_modular_gnn2_epoch16_valacc0.2760.pth"
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
    "vis_input_weights/hard/hard_200_fixed_modular_nlm/hard_200_fixed_modular_nlm_epoch27_valacc0.2520.pth"
    "vis_input_weights/hard/hard_200_fixed_modular_nlm/hard_200_fixed_modular_nlm_epoch24_valacc0.2452.pth"
    "vis_input_weights/hard/hard_200_fixed_modular_nlm/hard_200_fixed_modular_nlm_epoch16_valacc0.2485.pth"
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