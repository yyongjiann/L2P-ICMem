import os
import re
import numpy as np
import pandas as pd
import scipy.stats as stats

# Directory where log files are stored
directory = "results/baseline_2_exp"  # Change this to your actual directory

# Define categories
categories = ["low", "mixed", "high"]
std_data = {category: {} for category in categories}  # Store accuracy values per task
task_acc_data = {category: {} for category in categories}  # Store accuracy values per task
forgetting_data = {category: {} for category in categories}  # Store forgetting values per task

acc_error_bounds = {category: {} for category in categories} # Store error bounds per task
std_error_bounds = {category: {} for category in categories}  # Store error bounds per task
forgetting_error_bounds = {category: {} for category in categories}  # Store error bounds per task


def extract_acc_at_1(file_path):
    """
    Extracts Acc@1 values from a log file that contains:
    '[Average accuracy till taskN] Acc@1: X.XXXX'
    """
    acc_at_1_results = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"\[Average accuracy till task(\d+)\]\s+Acc@1:\s+([\d\.]+)", line)
            if match:
                task_number = int(match.group(1))
                acc_at_1 = float(match.group(2))
                acc_at_1_results.append((task_number, acc_at_1))
    return acc_at_1_results

def extract_std(file_path):
    """
    Extracts Acc@1 values from a log file that contains:
    '[Average accuracy till taskN] Acc@1: X.XXXX'
    """
    std_results = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"Standard Deviation of Similarities for task (\d+): ([0-9.]+)", line)
            if match:
                task_number = int(match.group(1))
                std = float(match.group(2))
                std_results.append((task_number, std))
    return std_results

def extract_forgetting(file_path):
    """
    Extracts Forgetting values from a log file that contains:
    '[Average accuracy till taskN] Acc@1: X.XXXX Acc@5: X.XXXX Loss: X.XXXX Forgetting: X.XXXX'
    """
    forgetting_results = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"\[Average accuracy till task(\d+)\]\s+Acc@1:\s+[\d\.]+\s+Acc@5:\s+[\d\.]+\s+Loss:\s+[\d\.]+\s+Forgetting:\s+([\d\.]+)", line)
            if match:
                task_number = int(match.group(1))
                forgetting = float(match.group(2))
                forgetting_results.append((task_number, forgetting))
    return forgetting_results

def compute_error_bound(data, confidence=0.90):
    """
    Compute the margin of error (error bound) for a given dataset using t-distribution.

    Args:
        data (list or np.array): List of numerical values (e.g., accuracy values).
        confidence (float): Confidence level (default: 95%).

    Returns:
        error_bound (float): The margin of error.
    """
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation (ddof=1)
    
    # Get the t-score for n-1 degrees of freedom
    t_score = stats.t.ppf((1 + confidence) / 2, df=n-1)

    # Compute the margin of error (error bound)
    error_bound = t_score * (std_dev / np.sqrt(n))
    
    return error_bound




# Process all relevant files
for filename in os.listdir(directory):
    for category in categories:
        if filename.startswith(category) and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            acc_results = extract_acc_at_1(file_path)
            std_results = extract_std(file_path)
            forgetting_results = extract_forgetting(file_path)

            for task, std in std_results:
                if task not in std_data[category]:
                    std_data[category][task] = []
                std_data[category][task].append(std)

            for task, acc in acc_results:
                if task not in task_acc_data[category]:
                    task_acc_data[category][task] = []
                task_acc_data[category][task].append(acc)

            for task, forgetting in forgetting_results:
                if task not in forgetting_data[category]:
                    forgetting_data[category][task] = []
                forgetting_data[category][task].append(forgetting)


average_std_data = {category: {} for category in categories}
for category, task_data in std_data.items():
    for task, std_list in task_data.items():
        average_std_data[category][task] = np.mean(std_list)
        error_bound = compute_error_bound(std_list)
        std_error_bounds[category][task] = error_bound

# Convert to DataFrame for better visualization
df_std = pd.DataFrame(average_std_data).sort_index()
df_std.index.name = "Task Number"

# Add error bounds to the DataFrame
df_error = pd.DataFrame(std_error_bounds).sort_index()
df_error.index.name = "Task Number"

print(df_std)
print(df_error)

# Compute averages for each category
average_acc_data = {category: {} for category in categories}
for category, task_data in task_acc_data.items():
    for task, acc_list in task_data.items():
        # Compute Error Bound
        error_bound = compute_error_bound(acc_list)
        acc_error_bounds[category][task] = error_bound
        average_acc_data[category][task] = np.mean(acc_list)

# Convert to DataFrame for better visualization
df_avg = pd.DataFrame(average_acc_data).sort_index()
df_avg.index.name = "Task Number"

# Add error bounds to the DataFrame
df_error = pd.DataFrame(acc_error_bounds).sort_index()
df_error.index.name = "Task Number"

# Display the result
print(df_avg)
print(df_error)

average_forgetting_data = {category: {} for category in categories}
for category, task_data in forgetting_data.items():
    for task, forgetting_list in task_data.items():
        # Compute Error Bound
        error_bound = compute_error_bound(forgetting_list)
        forgetting_error_bounds[category][task] = error_bound
        average_forgetting_data[category][task] = np.mean(forgetting_list)

# Convert to DataFrame for better visualization
df_for = pd.DataFrame(average_forgetting_data).sort_index()
df_for.index.name = "Task Number"

# Add error bounds to the DataFrame
df_error = pd.DataFrame(forgetting_error_bounds).sort_index()
df_error.index.name = "Task Number"

# Display the result
print(df_for)
print(df_error)
