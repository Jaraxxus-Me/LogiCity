checkpoints=(
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.1/transfer_e2e_gnn_random_0.1_epoch24_valacc0.4004.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.1/transfer_e2e_gnn_random_0.1_epoch25_valacc0.3671.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.1/transfer_e2e_gnn_random_0.1_epoch12_valacc0.3588.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.8/transfer_e2e_gnn_random_0.8_epoch29_valacc0.4728.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.8/transfer_e2e_gnn_random_0.8_epoch7_valacc0.4133.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.8/transfer_e2e_gnn_random_0.8_epoch28_valacc0.4081.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.05/transfer_e2e_gnn_random_0.05_epoch29_valacc0.3636.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.05/transfer_e2e_gnn_random_0.05_epoch24_valacc0.3540.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.05/transfer_e2e_gnn_random_0.05_epoch26_valacc0.3496.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.5/transfer_e2e_gnn_random_0.5_epoch27_valacc0.4287.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.5/transfer_e2e_gnn_random_0.5_epoch23_valacc0.4284.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.5/transfer_e2e_gnn_random_0.5_epoch26_valacc0.4039.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.2/transfer_e2e_gnn_random_0.2_epoch23_valacc0.4148.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.2/transfer_e2e_gnn_random_0.2_epoch22_valacc0.3967.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.2/transfer_e2e_gnn_random_0.2_epoch27_valacc0.3885.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_1.0/transfer_e2e_gnn_random_1.0_epoch29_valacc0.4684.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_1.0/transfer_e2e_gnn_random_1.0_epoch28_valacc0.4576.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_1.0/transfer_e2e_gnn_random_1.0_epoch25_valacc0.4469.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.01/transfer_e2e_gnn_random_0.01_epoch1_valacc0.2520.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.01/transfer_e2e_gnn_random_0.01_epoch0_valacc0.2518.pth"
    "vis_input_weights/hard/transfer_e2e_gnn_random_0.01/transfer_e2e_gnn_random_0.01_epoch2_valacc0.2513.pth"
)
# Loop over each checkpoint file
for ckpt in "${checkpoints[@]}"
do
    exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
    python3 tools/test_vis_input_e2e.py         --config config/tasks/Vis/ResNetGNN/hard_200_fixed_e2e.yaml         --ckpt "$ckpt"         --exp "$exp_name"         >> "log_vis/${exp_name}_transfer_test.log"
done
