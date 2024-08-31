""" 
This applies to the following per codon, multi-class (8) classification task
TASKS = [
    "ssp"
]
"""
import matplotlib.pyplot as plt
from collections import Counter
import torch
from datasets import load_dataset

dataset = load_dataset(
    "InstaDeepAI/true-cds-protein-tasks", name="ssp", trust_remote_code=True
)

# Plot histograms
plt.figure(figsize=(12, 5))

# Histogram for sequence lengths
plt.subplot(1, 2, 1)
plt.hist([len(x) for x in dataset['train']['sequence']], bins=30, color="blue", alpha=0.7)
plt.title("Histogram of Sequence Lengths (train split, {} ex)".format(len(dataset['train'])))
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")

# Histogram for label values
plt.subplot(1, 2, 2)
plt.hist([len(x) for x in dataset['CB513']['sequence']], bins=30, color="green", alpha=0.7)
plt.title("Histogram of Sequence Lengths (CB513/test split, {} ex)".format(len(dataset['CB513'])))
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")

plt.suptitle("Secondary Structure Prediction seq lengths for train and CB513 splits", fontsize=14)

plt.tight_layout()
plt.show()

# Sample data
label_sets = [
    dataset['train']['label'][i*10] for i in range(4)
]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each set of labels
for i, labels in enumerate(label_sets):
    counter = Counter(labels)
    label_values = list(counter.keys())
    label_counts = list(counter.values())
    
    ax = axes[i]
    ax.bar(label_values, label_counts, color='skyblue')
    ax.set_xlabel('Label')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Set {i+1}')
    ax.set_xticks(label_values)  # Ensure each label is shown on x-axis

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

# Sample data
label_sets = dataset['train']['label']

# Combine all label sets into a single list
all_labels = list(chain(*label_sets))

# Count the occurrences of each label
counter = Counter(all_labels)
label_values = list(counter.keys())
label_counts = list(counter.values())

# Plotting the aggregate picture
plt.figure(figsize=(10, 6))
plt.bar(label_values, label_counts, color='blue')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Aggregate Frequency of Labels')
plt.xticks(label_values)  # Ensure each label is shown on x-axis
plt.show()