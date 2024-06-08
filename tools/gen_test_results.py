import os
import re
from collections import defaultdict
import pandas as pd

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    metrics = {}
    metrics['file'] = file_path
    metrics['testing_sample_avg_acc'] = float(re.search(r'Testing Sample Avg Acc:\s+([\d.]+)', content).group(1))
    metrics['action_weighted_acc'] = float(re.search(r'Action Weighted Acc:\s+([\d.]+)', content).group(1))
    return metrics

def main():
    folder_path = 'log_vis/transfer/nlm_2'  # Replace with your folder path
    log_files = [f for f in os.listdir(folder_path) if f.endswith('.log')]
    
    experiments = defaultdict(list)

    for log_file in log_files:
        match = re.match(r'(.+?)_(\d+\.\d+)_(\d+)_epoch\d+_valacc[\d.]+\.pth_transfer_test\.log', log_file)
        if match:
            experiment_base_name = match.group(1)
            parameter_value = match.group(2)
            experiment_name = f"{experiment_base_name}_{parameter_value}"
            log_file_path = os.path.join(folder_path, log_file)
            metrics = parse_log_file(log_file_path)
            metrics['parameter_value'] = parameter_value
            experiments[experiment_base_name].append(metrics)

    results = defaultdict(dict)

    for experiment_base_name, metrics_list in experiments.items():
        for metrics in metrics_list:
            param_value = metrics['parameter_value']
            if param_value not in results[experiment_base_name]:
                results[experiment_base_name][param_value] = {
                    "Avg Acc": metrics['testing_sample_avg_acc'],
                    "wAvg Acc": metrics['action_weighted_acc']
                }
            else:
                if metrics['action_weighted_acc'] > results[experiment_base_name][param_value]["wAvg Acc"]:
                    results[experiment_base_name][param_value] = {
                        "Avg Acc": metrics['testing_sample_avg_acc'],
                        "wAvg Acc": metrics['action_weighted_acc']
                    }

    # Create DataFrame
    data = defaultdict(dict)
    for experiment_base_name, param_dict in results.items():
        for param_value, metrics in param_dict.items():
            data[(param_value, "Avg Acc")][experiment_base_name] = metrics["Avg Acc"]
            data[(param_value, "wAvg Acc")][experiment_base_name] = metrics["wAvg Acc"]

    df = pd.DataFrame(data).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Parameter Value", "Metric"])
    df = df.sort_index()

    # Transpose DataFrame
    df = df.T

    # Save to Excel
    df.to_excel('results.xlsx')

if __name__ == '__main__':
    main()
