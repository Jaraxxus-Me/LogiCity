import os
import re

# Base directory containing the checkpoint folders
base_dir = "vis_input_weights/hard"
run = "transfer_modular_nlm_fixed"

# Regular expression to match the folders
folder_pattern = re.compile(r"{}_\d+\.\d+_\d+".format(run))

# Regular expression to extract the valacc value from filenames
valacc_pattern = re.compile(r"valacc([\d\.]+).pth")

# List to hold all top checkpoints
all_top_checkpoints = []

# Loop through the directories in the base directory
for folder_name in os.listdir(base_dir):
    if folder_pattern.match(folder_name):
        folder_path = os.path.join(base_dir, folder_name)
        checkpoints = []
        
        # Loop through the files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".pth"):
                match = valacc_pattern.search(filename)
                if match:
                    valacc = float(match.group(1))
                    checkpoints.append((filename, valacc))
        
        # Sort the checkpoints by valacc in descending order
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top 3 checkpoints
        top_3_checkpoints = checkpoints[:3]
        
        # Add the top 3 checkpoints to the main list
        for filename, valacc in top_3_checkpoints:
            all_top_checkpoints.append(os.path.join(folder_path, filename))

# Create the shell script content
shell_script = "checkpoints=(\n"

# Add the top checkpoints to the shell script
for ckpt in all_top_checkpoints:
    shell_script += f'    "{ckpt}"\n'

# Add the remaining part of the shell script
if 'e2e' in run:
    shell_script += """)
    # Loop over each checkpoint file
    for ckpt in "${checkpoints[@]}"
    do
        exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
        python3 tools/test_vis_input_e2e.py \
            --config config/tasks/Vis/ResNetNLM/hard_200_fixed_e2e.yaml \
            --ckpt "$ckpt" \
            --exp "$exp_name" \
            >> "log_vis/${exp_name}_transfer_test.log"
    done
    """
else:
    shell_script += """)
    # Loop over each checkpoint file
    for ckpt in "${checkpoints[@]}"
    do
        exp_name=$(basename "$ckpt" | cut -d'_' -f1-9)  # This extracts the part of the filename without extension
        python3 tools/test_vis_input_mod.py \
            --config config/tasks/Vis/ResNetNLM/hard_200_fixed_modular.yaml \
            --ckpt "$ckpt" \
            --exp "$exp_name" \
            >> "log_vis/${exp_name}_transfer_test.log"
    done
    """

# Save the shell script to a file
with open("run_top_checkpoints_{}.sh".format(run), "w") as f:
    f.write(shell_script)

print("Shell script created: run_top_checkpoints.sh")
