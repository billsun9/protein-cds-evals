# %%
# Imports and device setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
hf_path = "zhihan1996/DNABERT-2-117M"
pretrained_model = AutoModel.from_pretrained(
    hf_path, config=BertConfig.from_pretrained(hf_path), trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)


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


# FFN and Combined Model classes
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = (
            alpha  # Weight for the SmoothL1Loss, (1-alpha) is the weight for MSELoss
        )

    def forward(self, outputs, targets):
        l1_loss = self.smooth_l1_loss(outputs, targets)
        mse_loss = self.mse_loss(outputs, targets)
        combined_loss = self.alpha * l1_loss + (1 - self.alpha) * mse_loss
        return combined_loss


class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.ffn = FFN()

    def forward(self, input_ids, attention_mask):
        x = self.pretrained_model(input_ids, attention_mask=attention_mask)[0]
        x = torch.mean(x, dim=1)
        x = self.ffn(x)
        return x


# Updated TupleDataset class



# %%
def plot_actual_vs_predicted(y_true, y_pred, desc, n=100):
    y_true = y_true.cpu().numpy()[:n]
    y_pred = y_pred.cpu().numpy()[:n]

    plt.figure(figsize=(5, 3))
    plt.plot(y_true, label="Actual", marker="o", rasterized=True)
    plt.plot(y_pred, label="Predicted", marker="x", rasterized=True)
    plt.xlabel("Sample")
    plt.ylabel("Fluorescence")
    plt.title(f"Actual vs Predicted Fluorescence ({desc})")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig("imgs/{}.png".format(desc))


def train_and_validate(model, train_dataloader, val_dataloader, num_epochs=25):
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        y_true_all = []
        y_pred_all = []
        for input_ids, attention_mask, y in train_dataloader:
            input_ids, attention_mask, y = (
                input_ids.to(device),
                attention_mask.to(device),
                y.to(device).float(),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze()

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            y_true_all.extend(y.detach().cpu().numpy())
            y_pred_all.extend(outputs.detach().cpu().numpy())

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        plot_actual_vs_predicted(
            torch.tensor(y_true_all), torch.tensor(y_pred_all), "train epoch {}".format(epoch)
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        y_true_all = []
        y_pred_all = []
        with torch.no_grad():
            for input_ids, attention_mask, y in val_dataloader:
                input_ids, attention_mask, y = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    y.to(device).float(),
                )
                outputs = model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, y)
                val_loss += loss.item()

                # Collecting all true and predicted values for plotting
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(outputs.cpu().numpy())

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

        # Plotting after every epoch
        plot_actual_vs_predicted(
            torch.tensor(y_true_all), torch.tensor(y_pred_all), "val epoch {}".format(epoch)
        )


# Testing loop
def test(model, test_dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input_ids, attention_mask, y in test_dataloader:
            input_ids, attention_mask, y = (
                input_ids.to(device),
                attention_mask.to(device),
                y.to(device).float(),
            )
            outputs = model(input_ids, attention_mask).squeeze()
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(outputs.detach().cpu().numpy())

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Test MSE: {mse:.4f}, Test R2 Score: {r2:.4f}")
    # plot_actual_vs_predicted(torch.tensor(y_true), torch.tensor(y_pred), "test")


# %%
# Dataloader setup
dls = get_dataloaders("fluorescence")
train_dataloader = dls["train"]
val_dataloader = dls["val"]
test_dataloader = dls["test"]

# Instantiate model, train, test, and save
model = CombinedModel().to(device)
model = nn.DataParallel(model)
train_and_validate(model, train_dataloader, val_dataloader, num_epochs=50)
test(model, test_dataloader)
# torch.save(model.state_dict(), "final_model_xdd.pth")
# %%
