import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class TupleDataset(Dataset):
    def __init__(self, original_dataset, tokenizer, max_length=256):
        """Creates Dataset of (input_ids, attention_mask, label) from original Instadeep Protein Dataset

        Args:
            original_dataset (String): Dataset
            tokenizer (Tokenizer): May need to override with proper tokenization scheme
            max_length (int, optional): _description_. Defaults to 256.
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


def get_dataloaders(dataset_name):
    dataset = load_dataset(
        "InstaDeepAI/true-cds-protein-tasks", name=dataset_name, trust_remote_code=True
    )

    train_dset = TupleDataset(dataset["train"].select(range(100)), tokenizer)
    val_dset = TupleDataset(dataset["validation"].select(range(100)), tokenizer)
    test_dset = TupleDataset(dataset["test"].select(range(100)), tokenizer)

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
