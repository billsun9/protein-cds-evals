# %%
# Imports and device setup
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

from layers import FFN
from utils import get_dataloaders
from train_and_test import train_and_validate, tes
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
hf_path = "zhihan1996/DNABERT-2-117M"
pretrained_model = AutoModel.from_pretrained(
    hf_path, config=BertConfig.from_pretrained(hf_path), trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)


# %%
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.ffn = FFN()

    def forward(self, input_ids, attention_mask):
        """Simply take mean embedding along sequence length.
        Input: (batch of input_ids, batch of attn_masks) -> 
        Post pretrained_model: (batch size, seq len, embedding_dim) -> 
        Post torch.mean: (batch_size, embedding_dim) -> 
        Post FFN: (batch_size, 1)
        """
        x = self.pretrained_model(input_ids, attention_mask=attention_mask)[0]
        x = torch.mean(x, dim=1)
        x = self.ffn(x)
        return x


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
