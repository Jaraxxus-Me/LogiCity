checkpoints=(
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.1/transfer2_modular_gnn_random_0.1_epoch1_valacc0.4069.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.1/transfer2_modular_gnn_random_0.1_epoch3_valacc0.3772.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.1/transfer2_modular_gnn_random_0.1_epoch4_valacc0.3675.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.05/transfer2_modular_gnn_random_0.05_epoch3_valacc0.3842.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.05/transfer2_modular_gnn_random_0.05_epoch2_valacc0.3458.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.05/transfer2_modular_gnn_random_0.05_epoch5_valacc0.3060.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.2/transfer2_modular_gnn_random_0.2_epoch8_valacc0.2959.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.2/transfer2_modular_gnn_random_0.2_epoch17_valacc0.2912.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.2/transfer2_modular_gnn_random_0.2_epoch5_valacc0.2907.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.8/transfer2_modular_gnn_random_0.8_epoch28_valacc0.3128.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.8/transfer2_modular_gnn_random_0.8_epoch16_valacc0.3050.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.8/transfer2_modular_gnn_random_0.8_epoch25_valacc0.3034.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.5/transfer2_modular_gnn_random_0.5_epoch26_valacc0.3022.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.5/transfer2_modular_gnn_random_0.5_epoch18_valacc0.2973.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.5/transfer2_modular_gnn_random_0.5_epoch23_valacc0.2944.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_1.0/transfer2_modular_gnn_random_1.0_epoch27_valacc0.3099.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_1.0/transfer2_modular_gnn_random_1.0_epoch19_valacc0.3093.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_1.0/transfer2_modular_gnn_random_1.0_epoch25_valacc0.3050.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.01/transfer2_modular_gnn_random_0.01_epoch14_valacc0.3820.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.01/transfer2_modular_gnn_random_0.01_epoch16_valacc0.3722.pth"
    "vis_input_weights/hard/transfer2_modular_gnn_random_0.01/transfer2_modular_gnn_random_0.01_epoch23_valacc0.3694.pth"
)
# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py         --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e.yaml         --ckpt "$ckpt"         --exp "$exp_name"         >> "log_vis/${exp_name}_transfer_test.log"
done
