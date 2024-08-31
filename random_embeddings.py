# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer # , AutoModelForMaskedLM

from losses import CombinedLoss
from layers import FFN
from utils import get_dataloaders
from train_and_test import train_and_validate

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# see https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
hf_path = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
# pretrained_model = AutoModelForMaskedLM.from_pretrained(hf_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
ffn = FFN(in_shape=768)


# %%
class CombinedModel(nn.Module):
    def __init__(self):
        """Our Combined Model will take a tuple of (input_ids, attn_map), and create a random unif[0,1] vector of shape (768,).
        This is then saved in embedding_map (akin to a frozen pretrained model)
        """
        super().__init__()
        self.embedding_map = {}
        self.ffn = ffn

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        device = input_ids.device  # Ensure all operations are on the correct device
        embeddings = []

        for i in range(batch_size):
            # Convert the individual input_ids and attention_mask to tuples for hashability
            input_tuple = tuple(input_ids[i].cpu().numpy().flatten())
            attention_mask_tuple = tuple(attention_mask[i].cpu().numpy().flatten())
            
            # Create a combined key using both input_ids and attention_mask
            combined_key = (input_tuple, attention_mask_tuple)

            # Check if the embedding for this input already exists
            if combined_key not in self.embedding_map:
                # Generate a random embedding of size [768] and store it in the map
                self.embedding_map[combined_key] = torch.randn(768).to(device)

            # Retrieve the embedding from the map and append it to the list
            embeddings.append(self.embedding_map[combined_key].to(device))

        # Stack all embeddings to form a batch on the correct device
        embeddings = torch.stack(embeddings).to(device)

        # Pass the embeddings through the FFN
        x = self.ffn(embeddings)
        return x, embeddings
# %%
# Dataloader setup
dls = get_dataloaders("fluorescence", tokenizer=tokenizer)
train_dataloader = dls["train"]
val_dataloader = dls["val"]
test_dataloader = dls["test"]
criterion = CombinedLoss(alpha=0.8) # nn.MSELoss()


# Instantiate model, train, test, and save
model = CombinedModel().to(device)
model = nn.DataParallel(model)
# %%
model.train()
train_and_validate(
    model, train_dataloader, val_dataloader, criterion=criterion, num_epochs=10
)
# test(model, test_dataloader)
# torch.save(model.state_dict(), "final_model_xdd.pth")
# %%
