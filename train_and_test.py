# %%
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils import plot_actual_vs_predicted, plot_line, summarize_variance, summarize_cosine_similarity, summarize_euclidean_distance

# %%
def train_and_validate(
    model, train_dataloader, val_dataloader, criterion, lr=0.0005, num_epochs=25
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    var_per_epoch = []
    cos_per_epoch = []
    euc_per_epoch = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        y_true_all, y_pred_all = [], []
        var, cos, euc = [], [], []
        for input_ids, attention_mask, y in train_dataloader:
            input_ids, attention_mask, y = (
                input_ids.to(device),
                attention_mask.to(device),
                y.to(device).float(),
            )

            optimizer.zero_grad()
            outputs, embedding = model(input_ids, attention_mask)
            outputs = outputs.squeeze()

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            y_true_all.extend(y.detach().cpu().numpy())
            y_pred_all.extend(outputs.detach().cpu().numpy())
            var.append(summarize_variance(embedding))
            cos.append(summarize_cosine_similarity(embedding))
            euc.append(summarize_euclidean_distance(embedding))

        train_loss /= len(train_dataloader)
        var_ = sum(var) / len(var)
        cos_ = sum(cos) / len(cos)
        euc_ = sum(euc) / len(euc)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Avg Var: {var_:.8f}, Avg Cos Sim: {cos_:.8f}, Avg Euc Dist: {euc_:.8f}")
        var_per_epoch.append(var_)
        cos_per_epoch.append(cos_)
        euc_per_epoch.append(euc_)

        plot_actual_vs_predicted(
            torch.tensor(y_true_all),
            torch.tensor(y_pred_all),
            "train epoch {}".format(epoch),
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
                outputs, embedding = model(input_ids, attention_mask)
                outputs = outputs.squeeze()
                loss = criterion(outputs, y)
                val_loss += loss.item()

                # Collecting all true and predicted values for plotting
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(outputs.cpu().numpy())

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")

        # Plotting after every epoch
        plot_actual_vs_predicted(
            torch.tensor(y_true_all),
            torch.tensor(y_pred_all),
            "val epoch {}".format(epoch),
        )
    plot_line(var_per_epoch, "avg_var")
    plot_line(cos_per_epoch, "avg_cos_sim")
    plot_line(euc_per_epoch, "avg_euc_dist")
# will not work yet
def test_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
