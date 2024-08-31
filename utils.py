import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from datasets import load_dataset
import matplotlib.pyplot as plt

N_SAMPLES = 100

class TupleDatasetDNABERT2(Dataset):
    def __init__(self, original_dataset, tokenizer, max_length=256):
        """Creates Dataset of (input_ids, attention_mask, label) from original Instadeep Protein Dataset

        Args:
            original_dataset (Dataset)
            tokenizer (Tokenizer): Only works for DNABERT2 (i.e. BPE)
            max_length (int, optional): DNABERT2 max is 512. Defaults to 256.
        """
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        encoding = self.tokenizer(
            item["sequence"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(item["label"], dtype=torch.float)
        return input_ids, attention_mask, label
    

class TupleDatasetNT(Dataset):
    def __init__(self, original_dataset, tokenizer, max_length=512):
        """Creates Dataset of (input_ids, attention_mask, label) from original Instadeep Protein Dataset

        Args:
            original_dataset (Dataset)
            tokenizer (Tokenizer): Only works for NT (aparently 6mer + special tokens)
            max_length (int, optional): max is 2048. Default is 512
        """
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]

        input_ids = self.tokenizer.batch_encode_plus(item, return_tensors="pt", padding="max_length", max_length = self.max_length)["input_ids"]
        encoding = self.tokenizer(
            item["sequence"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(item["label"], dtype=torch.float)
        return input_ids, attention_mask, label


def get_dataloaders(dataset_name, tokenizer):
    dataset = load_dataset(
        "InstaDeepAI/true-cds-protein-tasks", name=dataset_name, trust_remote_code=True
    )

    # DNABERT and Nucleotide Transformer have similar call syntax, but we separate for clarity
    if type(tokenizer) == transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
        train_dset = TupleDatasetDNABERT2(dataset["train"].select(range(100)), tokenizer)
        val_dset = TupleDatasetDNABERT2(dataset["validation"].select(range(100)), tokenizer)
        test_dset = TupleDatasetDNABERT2(dataset["test"].select(range(100)), tokenizer)
    # Nucleotide Transformer
    elif type(tokenizer) == transformers.models.esm.tokenization_esm.EsmTokenizer:
        train_dset = TupleDatasetNT(dataset["train"].select(range(100)), tokenizer)
        val_dset = TupleDatasetNT(dataset["validation"].select(range(100)), tokenizer)
        test_dset = TupleDatasetNT(dataset["test"].select(range(100)), tokenizer)
    else:
        raise Exception("tokenizer is wack")

    # May need to alter num_workers based on config
    return {
        "train": DataLoader(
            train_dset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        ),
        "val": DataLoader(
            val_dset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        ),
        "test": DataLoader(
            test_dset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        ),
    }

def plot_actual_vs_predicted(y_true, y_pred, desc, n=100):
    """Plot the predictions of GT Labels vs Model Predictions for first n samples. Uses small plot + rasterized

    Args:
        desc (String): Title of plot
        n (int, optional): First n samples. Defaults to 100.
    """
    y_true = y_true.cpu().numpy()[:n]
    y_pred = y_pred.cpu().numpy()[:n]

    plt.figure(figsize=(5, 3))
    plt.plot(y_true, label="Actual", marker="o", rasterized=True)
    plt.plot(y_pred, label="Predicted", marker="x", rasterized=True)
    plt.xlabel("Sample")
    plt.ylabel("Regression Value")
    plt.title(f"Actual vs Predicted ({desc})")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig("imgs/{}.png".format(desc))

def plot_line(data, desc):
    plt.figure(figsize=(5, 3))
    plt.plot(data, marker="o", rasterized=True)
    plt.xlabel("Epochs")
    plt.ylabel(f"{desc}")
    plt.title(f"Change in training {desc} over epochs")
    plt.grid(True)
    plt.show()

def plot_grads(model):
    """Useful for gradient clipping + viewing vanishing/exploding grads
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            plt.hist(param.grad.cpu().numpy().flatten(), bins=50)
            plt.title(f"Gradient histogram for {name}")
            plt.show()

def summarize_variance(embeddings):
    """
    Calculate the average variance of embeddings across all features.

    Args:
        embeddings (torch.Tensor)
    Returns:
        float: The average variance across all features
    """
    feature_variances = torch.var(embeddings, dim=0)
    average_variance = torch.mean(feature_variances).item()

    return average_variance

def summarize_euclidean_distance(embeddings):
    distances = torch.cdist(embeddings, embeddings, p=2)
    triu_indices = torch.triu_indices(distances.size(0), distances.size(1), offset=1)
    pairwise_distances = distances[triu_indices[0], triu_indices[1]]
    return pairwise_distances.mean().item()

def summarize_cosine_similarity(embeddings):
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    cosine_sim = torch.mm(norm_embeddings, norm_embeddings.t())
    triu_indices = torch.triu_indices(cosine_sim.size(0), cosine_sim.size(1), offset=1)
    pairwise_cosine_sim = cosine_sim[triu_indices[0], triu_indices[1]]
    return pairwise_cosine_sim.mean().item()