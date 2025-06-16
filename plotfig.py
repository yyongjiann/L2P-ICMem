import matplotlib.pyplot as plt
import numpy as np

# Task numbers
tasks = np.array([1, 2, 3, 4])

b1_acc_low = np.array([99.200000, 94.933333, 92.199993, 89.233333])
b1_acc_mixed = np.array([98.866667, 94.800000, 90.955553, 83.083333])
b1_acc_high = np.array([98.066667, 94.733333, 90.555553, 85.233333])

b1_acc_low_error = np.array([0.733292, 2.258959, 2.531912, 1.837991])
b1_acc_mixed_error = np.array([0.954962, 2.056898, 2.293155, 3.483260])
b1_acc_high_error = np.array([2.015301, 1.800435, 3.230798, 2.713499])

b1_sd_low = np.array([0.078133, 0.083960, 0.089333, 0.096320])
b1_sd_mixed = np.array([0.071860, 0.076780, 0.081853, 0.088393])
b1_sd_high = np.array([0.068233, 0.073553, 0.078727, 0.085627])

b1_sd_low_error = np.array([0.004383, 0.004746, 0.005642, 0.007003])
b1_sd_mixed_error = np.array([0.004996, 0.005365, 0.006437, 0.007861])
b1_sd_high_error = np.array([0.004974, 0.005407, 0.006655, 0.008263])

# Forgetting values (first column)
b1_forgetting_low = np.array([9.200000, 11.233333, 14.044440])
b1_forgetting_mixed = np.array([9.333333, 12.966667, 22.155560])
b1_forgetting_high = np.array([9.266667, 13.533333, 19.288887])

# Error values (first column)
b1_forgetting_low_error = np.array([4.325254, 3.889216, 2.511123])
b1_forgetting_mixed_error = np.array([3.838366, 3.594711, 4.741640])
b1_forgetting_high_error = np.array([3.700415, 5.043234, 3.710008])


b2_acc_low = np.array([99.767440, 96.162793, 95.860473, 94.158913])
b2_acc_mixed = np.array([99.76744, 95.96124, 96.40828, 94.61240])
b2_acc_high = np.array([99.767440, 95.759693, 97.023260, 95.976740])

b2_acc_low_error = np.array([0.218943, 1.236279, 0.977497, 1.971556])
b2_acc_mixed_error = np.array([0.218943, 1.348393, 1.097859, 1.829523])
b2_acc_high_error = np.array([0.218943, 1.864007, 0.759124, 1.049032])

b2_sd_low = np.array([0.189973, 0.201727, 0.214193, 0.231260])
b2_sd_mixed = np.array([0.189973, 0.201207, 0.213287, 0.229673])
b2_sd_high = np.array([0.189973, 0.200727, 0.212380, 0.228487])

b2_sd_low_error = np.array([0.009612, 0.010905, 0.013542, 0.017114])
b2_sd_mixed_error = np.array([0.009612, 0.010932, 0.013644, 0.017371])
b2_sd_high_error = np.array([0.009612, 0.010964, 0.013862, 0.017870])

b2_forgetting_low = np.array([7.379840, 5.751940, 7.462527])
b2_forgetting_mixed = np.array([7.720927, 4.542633, 6.589147])
b2_forgetting_high = np.array([8.093027, 4.085273, 5.059433])

# Error values (first column)
b2_forgetting_low_error = np.array([2.518158, 1.526454, 2.621927])
b2_forgetting_mixed_error = np.array([2.626779, 1.508368, 2.493329])
b2_forgetting_high_error = np.array([3.722957, 1.205494, 1.364672])



# Plot the data
plt.figure(figsize=(10, 6))

"""
plt.errorbar(tasks, b2_acc_low, yerr=b2_acc_low_error, capsize=10, fmt='o', linestyle='-', label="Low-Mem")
plt.errorbar(tasks, b2_acc_mixed, yerr=b2_acc_mixed_error, capsize=10, fmt='o', linestyle='-', label="Mixed-Mem")
plt.errorbar(tasks, b2_acc_high, yerr=b2_acc_high_error, capsize=10, fmt='o', linestyle='-', label="High-Mem")
"""

tasks = np.array([2, 3, 4])
plt.errorbar(tasks, b1_forgetting_low, yerr=b1_forgetting_low_error, capsize=10, fmt='o', linestyle='-', label="Low-Mem")
plt.errorbar(tasks, b1_forgetting_mixed, yerr=b1_forgetting_mixed_error, capsize=10, fmt='o', linestyle='-', label="Mixed-Mem")
plt.errorbar(tasks, b1_forgetting_high, yerr=b1_forgetting_high_error, capsize=10, fmt='o', linestyle='-', label="High-Mem")

# Labels and title
plt.xlabel("Task Number")
plt.ylabel("Forgetting")
plt.title("Task Number vs. Forgetting")
#plt.ylabel("Standard Deviation (SD)")
#plt.title("Task Number vs. SD")
plt.grid(True, linestyle='--', alpha=0.6)

#plt.xticks(tasks, labels=["1", "2", "3", "4"])
plt.xticks(tasks, labels=["2", "3", "4"])
plt.legend()
plt.grid(True)

plt.show()
