import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import sys
import re
import json
import time
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations

from eigentrust import eigentrust, row_normalize, compute_trust_matrix, compute_trust_matrix_ties
from data_utils import *

"""
Code for training BTD models with ternary comparison data, r in {0,1,2}, where judge responded with <criterion> comparisons.

Comparisons should be in the format [c, l, i, j, k, r]
"""

class Comparisons(Dataset):
    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c, l, i, j, k, r = self.data[idx]
        return torch.tensor(c, dtype=torch.long), \
               torch.tensor(i, dtype=torch.long), \
               torch.tensor(j, dtype=torch.long), \
               torch.tensor(k, dtype=torch.long), \
               torch.tensor(r, dtype=torch.float32)

class VectorBTD(nn.Module):
    def __init__(self, num_criteria, num_models, d):
        super().__init__()
        self.num_criteria = num_criteria
        self.num_models = num_models

        self.u = nn.Embedding(num_criteria * num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.log_lambda = nn.Embedding(num_criteria * num_models, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.log_lambda.weight, 0.0) # lambda initialized to 1

        # print(self.v.weight)

    def forward(self, c, i, j, k):
        judge = c * self.num_models + i
        u_i_c = self.u(judge)   # shape: (batch, d)
        v_j = self.v(j)     # shape: (batch, d)
        v_k = self.v(k)     # shape: (batch, d)

        score_j = torch.sum(u_i_c * v_j, dim=-1)
        score_k = torch.sum(u_i_c * v_k, dim=-1)

        log_lambda_i = self.log_lambda(judge).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
        return logits
    
    def get_prob(self, c, i, j, k):
        """
        get probs for a single set of indices c,i,j,k
        """
        judge = c * self.num_models + i
        u_i_c = self.u(judge)   # shape: (batch, d)
        v_j = self.v(j)     # shape: (batch, d)
        v_k = self.v(k)     # shape: (batch, d)

        score_j = torch.sum(u_i_c * v_j, dim=-1)
        score_k = torch.sum(u_i_c * v_k, dim=-1)

        log_lambda_i = self.log_lambda(judge).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = [tie_logit, score_j, score_k]
        return logits


def train_vector_bt(model, dataloader, lr, weight_decay, max_epochs, device, save_path=None, normalize=False, use_btd=False):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_btd:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCELoss()

    loss_history = []

    for epoch in range(1, max_epochs+1):
        total_loss = 0.0
        model.train()

        for c, i, j, k, r in dataloader:
            c = c.to(device)
            i = i.to(device)
            j = j.to(device)
            k = k.to(device)
            r = r.to(device)

            if use_btd:
                r = r.long()  # CrossEntropyLoss expects long tensor

                logits = model(c, i, j, k)
                loss = loss_fn(logits, r) # CE expects logits, unnormalized, as it has built in softmax
            else:
                p = model(i, j, k)
                loss = loss_fn(p, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                with torch.no_grad():
                    model.v.weight.data = F.normalize(model.v.weight.data, p=2, dim=1)

            total_loss += loss.item() * r.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)

        if len(loss_history) >= 10 and  np.average(np.abs(np.diff(loss_history[-10:]))) <= .0001:
            print('loss converged, breaking')
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3d}, Loss = {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.tight_layout()

    if save_path:
        loss_path = save_path+"training_loss.png"
        plt.savefig(loss_path)
        print(f"Loss plot saved to {loss_path}")

        model_path = save_path+"model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return loss_history


if __name__ == "__main__":
    
    USE_BTD = True
    NORMALIZE = False

    TEST = True
    TRUST = True

    NUM_CRITERIA = 8
    SEPARATE_CRITERIA = False

    path = 'transcript/20250726_000000/'#conservatism_grok/'

    dims = range(1,11)

    lr = 1e-3
    weight_decay = 0
    max_epochs = 1000
    batch_size = 32
    device = 'cpu'

    filepath = path + 'evaluations.json'
    cleaned_filepath = path + 'evaluations_cleaned.json'

    data = []
    with open(filepath, 'r') as file:
        data.extend(json.load(file))

    comparisons, data_cleaned = extract_comparisons_with_ties_criteria(data, num_criteria=NUM_CRITERIA)
    comparisons = handle_inconsistencies_with_ties_criteria(comparisons)

    # comparisons = json.load(open(path + 'comparisons_human_shreyas.json', 'r'))
    # comparisons = [i[:2] + [6] + i[3:] for i in comparisons]
    # comparisons = [i for i in comparisons if i[2] not in [5,6,7] and i[3] not in [5,6,7] and i[4] not in [5,6,7]]

    print("Loaded data has length", len(data))
    print("Cleaned & criterion-separated data has length", len(data_cleaned))
    print("Formed", len(comparisons), "comparisons after handling inconsistencies\n")

    if not SEPARATE_CRITERIA:
        comparisons = [[0] + i[1:] for i in comparisons]  # make every datapoint associated with criterion 0
    
    num_models = len(set([i[2] for i in comparisons ] + [i[3] for i in comparisons] + [i[4] for i in comparisons])) # count up all unique judges and evaluees
    num_criteria = len(set([i[0] for i in comparisons]))

    print("Number of models:", num_models, ", Number of criteria:", num_criteria)

    for d in dims:

        out_dir = path + f"btd_d{d}/"
        os.makedirs(out_dir, exist_ok=True)

        model = VectorBTD(num_criteria, num_models, d)

        print(f"Ready to train model (d={d}):\n")

        if TEST:
            train_comps, test_comps = train_test_split(
                comparisons,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )
            train_loader = DataLoader(Comparisons(train_comps), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(Comparisons(test_comps), batch_size=batch_size, shuffle=False)

            loss_history = train_vector_bt(
                model,
                train_loader,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                device=device,
                save_path=out_dir,
                normalize=NORMALIZE,
                use_btd=USE_BTD,
            )

            print("Evaluating on test set...\n")
            model.eval()

            if USE_BTD:
                loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = nn.BCELoss()

            total_test_loss = 0.0
            loss_matrix = np.zeros((num_criteria * num_models, num_models, num_models))
            count_matrix = np.zeros((num_criteria * num_models, num_models, num_models))

            with torch.no_grad():
                for c, i, j, k, r in test_loader:
                    c, i, j, k = c.to(device), i.to(device), j.to(device), k.to(device)
                    r = r.to(device)

                    if USE_BTD:
                        r = r.long()
                        logits = model(c, i, j, k)
                        loss = loss_fn(logits, r)
                        per_sample_loss = F.cross_entropy(logits, r, reduction="none").cpu().numpy()
                    else:
                        p = model(i, j, k)
                        loss = loss_fn(p, r)
                        per_sample_loss = F.binary_cross_entropy(p, r, reduction="none").cpu().numpy()

                    total_test_loss += loss.item() * r.size(0)

                    c_np = c.cpu().numpy()
                    i_np = i.cpu().numpy()
                    j_np = j.cpu().numpy()
                    k_np = k.cpu().numpy()
                    for n in range(r.size(0)):
                        row = c_np[n] * num_models + i_np[n]
                        col_j = j_np[n]
                        col_k = k_np[n]
                        loss_matrix[row][col_j][col_k] += per_sample_loss[n]
                        count_matrix[row][col_j][col_k] += 1

            avg_loss_matrix = np.divide(
                loss_matrix,
                count_matrix,
                out=np.zeros_like(loss_matrix),
                where=(count_matrix != 0),
            )

            matrix_str = np.array2string(
                np.round(avg_loss_matrix, 4),
                separator=', ',
                max_line_width=np.inf,
                threshold=np.inf,
                formatter={'float_kind': lambda x: f"{x:0.4f}"}
            )

            counts_str = np.array2string(
                np.round(count_matrix, 4),
                separator=', ',
                max_line_width=np.inf,
                threshold=np.inf,
                formatter={'float_kind': lambda x: f"{x:0.4f}"}
            )

            avg_test_loss = total_test_loss / len(test_loader.dataset)

            log_path = out_dir + "log_train.txt"
            with open(log_path, "w") as f:
                f.write(f"dataset = {path}\n")
                f.write(f"train_datasize = {len(train_comps)}\n")
                f.write(f"test_datasize = {len(test_comps)}\n")
                f.write(f"batch_size = {batch_size}\n\n")
                f.write(f"model = {model.__class__.__name__}\n")
                f.write(f"num_models = {num_models}\n")
                f.write(f"normalize = {NORMALIZE}\n")
                f.write(f"dim = {d}\n\n")
                f.write(f"lr = {lr}\n")
                f.write(f"weight_decay = {weight_decay}\n")
                f.write(f"epochs = {max_epochs}\n\n")
                f.write(f"Min Train Loss = {np.round(min(loss_history),4)}\n")
                f.write(f"Test Loss = {np.round(avg_test_loss,4)}\n\n")
                f.write(f"Test Loss Matrix:\n{matrix_str}\n\n")
                f.write(f"Test Counts Matrix:\n{counts_str}")

            print(f"Training log written to {log_path}")

        if TRUST:
            print("Computing eigentrust scores...")

            if USE_BTD:
                T = compute_trust_matrix_ties(model, device)
                t = eigentrust(T, alpha=0)

                trust_path = out_dir + "eigentrust.txt"
                with open(trust_path, "w") as f:
                    T_str = np.array2string(
                        T.cpu().numpy(),
                        separator=', ',
                        max_line_width=np.inf,
                        threshold=np.inf,
                        formatter={'float_kind': lambda x: f"{x:0.4f}"}
                    )

                    t_str = np.array2string(
                        t.cpu().numpy(),
                        separator=', ',
                        max_line_width=np.inf,
                        threshold=np.inf,
                        formatter={'float_kind': lambda x: f"{x:0.4f}"}
                    )
                    f.write("Trust matrix:\n")
                    f.write(T_str + "\n")
                    f.write("EigenTrust scores:\n")
                    f.write(t_str + "\n")

            else:
                S = compute_trust_matrix(model, device)
                C = row_normalize(S)
                t = eigentrust(C, alpha=0)

                trust_path = out_dir + "eigentrust.txt"
                with open(trust_path, "w") as f:
                    f.write("Trust matrix:\n")
                    f.write(str(S.cpu().numpy()) + "\n")
                    f.write("Row-normalized:\n")
                    f.write(str(C.cpu().numpy()) + "\n")
                    f.write("EigenTrust scores:\n")
                    f.write(str(t.cpu().numpy()) + "\n")

            print(f"Eigentrust scores written to {trust_path}")

        print(f"Finished training dimension d = {d}\n")