import os

def remove_ds_store(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

# Example usage:
folder_to_clean = "imgs/all"
remove_ds_store(folder_to_clean)