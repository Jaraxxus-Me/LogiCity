import numpy as np

def combine_npy_files(file1, file2, file3, output_file):
    # Load the .npy files
    array1 = np.load(file1)
    array2 = np.load(file2)
    array3 = np.load(file3)
    
    # Combine the arrays
    combined_array = np.concatenate((array1, array2, array3), axis=0)
    
    # Save the combined array to a new .npy file
    np.save(output_file, combined_array)
    
    return combined_array

# Example usage
file1 = 'log_vis/gpt/res_list_gpt4t_5shot_gp_0_4800.npy'
file2 = 'log_vis/gpt/res_list_gpt4t_5shot_gp_4800_9600.npy'
file3 = 'log_vis/gpt/res_list_gpt4t_5shot_gp_9600_-1.npy'
output_file = 'log_vis/gpt/res_list_gpt4t_5shot_gp_0_-1.npy'

combined_array = combine_npy_files(file1, file2, file3, output_file)
print("Combined array saved to:", output_file)