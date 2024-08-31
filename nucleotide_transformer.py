# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from losses import CombinedLoss
from layers import FFN
from utils import get_dataloaders
from train_and_test import train_and_validate

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# see https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
hf_path = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
pretrained_model = AutoModelForMaskedLM.from_pretrained(hf_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
ffn = FFN(in_shape=512)


# %%
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.ffn = ffn

    def forward(self, input_ids, attention_mask):
        """Simply take mean embedding along sequence length.
        Input: (batch of input_ids, batch of attn_masks) ->
        Post pretrained_model: (batch size, seq len, embedding_dim) ->
        Post torch.mean: (batch_size, embedding_dim) ->
        Post FFN: (batch_size, 1)
        """
        embedding = self.pretrained_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )['hidden_states'][-1]
        embedding = torch.mean(embedding, dim=1)
        output = self.ffn(embedding)
        return output, embedding
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
    model, train_dataloader, val_dataloader, criterion=criterion, num_epochs=40
)
# test(model, test_dataloader)
# torch.save(model.state_dict(), "final_model_xdd.pth")
# %%
