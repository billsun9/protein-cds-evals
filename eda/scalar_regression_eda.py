""" 
This applies to the following scalar regression tasks
TASKS = [
    "beta_lactamase_complete",
    "beta_lactamase_unique",
    "stability",
    "melting_point",
    "fluorescence",
]
"""

# %%
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# %%
def vis(task_name, split="train"):
    """Loads and visualizes useful information about task_name, split"""
    dataset = load_dataset(
        "InstaDeepAI/true-cds-protein-tasks", name=task_name, trust_remote_code=True
    )

    dataloader = DataLoader(dataset[split], batch_size=1, shuffle=False)

    sequence_lengths = []
    label_values = []

    for batch in dataloader:
        sequence = batch["sequence"][0]
        label = batch["label"]
        sequence_lengths.append(len(sequence))  # Get length of the sequence
        label_values.append(label)  # Get the label value

    label_values = [round(val.numpy()[0], 5) for val in label_values]

    # Calculate statistics for sequence lengths
    avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
    min_seq_len = min(sequence_lengths)
    max_seq_len = max(sequence_lengths)

    # Calculate statistics for label values
    avg_label_val = sum(label_values) / len(label_values)
    min_label_val = min(label_values)
    max_label_val = max(label_values)

    print(f"Average sequence length: {avg_seq_len}")
    print(f"Min sequence length: {min_seq_len}")
    print(f"Max sequence length: {max_seq_len}")

    print(f"Average label value: {avg_label_val}")
    print(f"Min label value: {min_label_val}")
    print(f"Max label value: {max_label_val}")

    # Plot histograms
    plt.figure(figsize=(12, 5))

    # Histogram for sequence lengths
    plt.subplot(1, 2, 1)
    plt.hist(sequence_lengths, bins=30, color="blue", alpha=0.7)
    plt.title("Histogram of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")

    # Histogram for label values
    plt.subplot(1, 2, 2)
    plt.hist(label_values, bins=30, color="green", alpha=0.7)
    plt.title("Histogram of Label Values")
    plt.xlabel("Label Value")
    plt.ylabel("Frequency")

    plt.suptitle("{}: {}".format(task_name, split), fontsize=16)

    plt.tight_layout()
    plt.show()


# %%
TASKS = [
    "beta_lactamase_complete",
    "beta_lactamase_unique",
    # 'ssp',
    "stability",
    "melting_point",
    "fluorescence",
]

for TASK in TASKS:
    task_name = TASK
    print("------", task_name, "------")
    vis(task_name=task_name, split="train")
    vis(task_name=task_name, split="test")
# %%
dataset = load_dataset(
    "InstaDeepAI/true-cds-protein-tasks", name="ssp", trust_remote_code=True
)
# %%
for k in dataset.keys():
    for elem in dataset[k]:
        assert len(elem["sequence"]) // 3 == len(elem["label"]), "{}".format(elem)
# %%