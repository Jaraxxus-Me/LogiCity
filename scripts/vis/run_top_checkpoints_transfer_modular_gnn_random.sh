checkpoints=(
    "vis_input_weights/hard/transfer_modular_gnn_random_0.2/transfer_modular_gnn_random_0.2_epoch9_valacc0.3192.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.2/transfer_modular_gnn_random_0.2_epoch21_valacc0.3164.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.2/transfer_modular_gnn_random_0.2_epoch6_valacc0.3138.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.01/transfer_modular_gnn_random_0.01_epoch0_valacc0.2579.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.01/transfer_modular_gnn_random_0.01_epoch3_valacc0.2524.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.01/transfer_modular_gnn_random_0.01_epoch2_valacc0.2507.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.05/transfer_modular_gnn_random_0.05_epoch26_valacc0.3626.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.05/transfer_modular_gnn_random_0.05_epoch27_valacc0.3531.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.05/transfer_modular_gnn_random_0.05_epoch25_valacc0.3502.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.5/transfer_modular_gnn_random_0.5_epoch2_valacc0.4280.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.5/transfer_modular_gnn_random_0.5_epoch3_valacc0.3100.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.5/transfer_modular_gnn_random_0.5_epoch5_valacc0.2910.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.8/transfer_modular_gnn_random_0.8_epoch2_valacc0.3736.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.8/transfer_modular_gnn_random_0.8_epoch1_valacc0.3710.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.8/transfer_modular_gnn_random_0.8_epoch3_valacc0.3553.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_1.0/transfer_modular_gnn_random_1.0_epoch2_valacc0.3432.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_1.0/transfer_modular_gnn_random_1.0_epoch3_valacc0.3422.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_1.0/transfer_modular_gnn_random_1.0_epoch1_valacc0.3389.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.1/transfer_modular_gnn_random_0.1_epoch16_valacc0.3564.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.1/transfer_modular_gnn_random_0.1_epoch18_valacc0.3518.pth"
    "vis_input_weights/hard/transfer_modular_gnn_random_0.1/transfer_modular_gnn_random_0.1_epoch12_valacc0.3516.pth"
)
# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_mod.py         --config config/tasks/Vis/ResNetGNN/hard_200_fixed_modular2.yaml         --ckpt "$ckpt"         --exp "$exp_name"         >> "log_vis/${exp_name}_transfer_test.log"
done
