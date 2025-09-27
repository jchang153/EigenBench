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
Code for training BTD_length models with ternary comparison data, r in {0,1,2}, where judge responded with single <choice> comparisons.

Comparisons should be in the format [l, i, j, k, n_j, n_k, r]
"""

class Comparisons(Dataset):
    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        l, i, j, k, n_j, n_k, r = self.data[idx]
        return torch.tensor(i, dtype=torch.long), \
               torch.tensor(j, dtype=torch.long), \
               torch.tensor(k, dtype=torch.long), \
               torch.tensor(n_j, dtype=torch.long), \
               torch.tensor(n_k, dtype=torch.long), \
               torch.tensor(r, dtype=torch.float32)

class VectorBTD_length(nn.Module):
    def __init__(self, num_judges, num_evals, d):
        super().__init__()
        self.u = nn.Embedding(num_judges, d)
        self.v = nn.Embedding(num_evals, d)
        self.log_lambda = nn.Embedding(num_judges, 1)
        self.l = nn.Embedding(num_judges, 1)

        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.log_lambda.weight, 0.0) # lambda initialized to 1
        nn.init.constant_(self.l.weight, 0) # length parameter initialized to 0

    def forward(self, i, j, k, n_j, n_k):
        u_i = self.u(i)   # shape: (batch, d)
        v_j = self.v(j)   # shape: (batch, d)
        v_k = self.v(k)   # shape: (batch, d)

        # Calculate length ratios
        r_j = n_j / (n_j + n_k + 1)  # shape: (batch,)
        r_k = n_k / (n_j + n_k + 1)  # shape: (batch,)

        l_i = self.l(i).squeeze(-1) # shape: (batch,)
        
        # Calculate scores with length adjustment
        score_j = torch.sum(u_i * v_j, dim=-1) + l_i * r_j
        score_k = torch.sum(u_i * v_k, dim=-1) + l_i * r_k

        log_lambda_i = self.log_lambda(i).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
        return logits


def train_vector_bt(model, dataloader, lr, weight_decay, max_epochs, device, save_path=None, normalize=False):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    loss_history = []

    for epoch in range(1, max_epochs+1):
        total_loss = 0.0
        model.train()

        for i, j, k, n_j, n_k, r in dataloader:
            i = i.to(device)
            j = j.to(device)
            k = k.to(device)
            n_j = n_j.to(device)
            n_k = n_k.to(device)
            r = r.to(device)

            r = r.long()  # CrossEntropyLoss expects long tensor
            logits = model(i, j, k, n_j, n_k)
            loss = loss_fn(logits, r) # CE expects logits, unnormalized, as it has built in softmax

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
            print(f"Converged at epoch {epoch}, stopping training.")
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
    
    NORMALIZE = False
    TRUST = False

    path = 'transcript/20250730_000000/openai_rerun/'
    filepath = path + 'evaluations.json'

    data = []
    with open(filepath, 'r') as file:
        data.extend(json.load(file))

    comparisons, data_cleaned = extract_comparisons_with_lengths(data)
    comparisons = handle_inconsistencies_with_lengths(comparisons) # has the format [l, i, j, k, n_j, n_k, r]

    print("Loaded data has length", len(data))
    print("Cleaned data has length", len(data_cleaned))
    print("Formed", len(comparisons), "comparisons after handling inconsistencies\n")

    train_comps, test_comps = train_test_split(comparisons,test_size=0.2,random_state=42,shuffle=True)

    train_loader = DataLoader(Comparisons(train_comps), batch_size=32, shuffle=True)
    test_loader = DataLoader(Comparisons(test_comps), batch_size=32, shuffle=False)

    batch_size = 32
    num_judges = len(set([i[1] for i in comparisons]))
    num_evals = len(set([i[2] for i in comparisons]))
    d = 4

    lr = 1e-3
    weight_decay = 0
    max_epochs = 10000
    device = 'cpu'

    model = VectorBTD_length(num_judges, num_evals, d)


    loss_history = train_vector_bt(
        model, 
        train_loader,
        lr=lr, 
        weight_decay=weight_decay,
        max_epochs=max_epochs, 
        device=device,
        save_path=path,
        normalize=NORMALIZE,
    )

    print('Now evaluating the model on the test set\n')
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
        
    total_test_loss = 0.0
    loss_matrix = np.zeros((num_judges, num_evals, num_evals))
    count_matrix = np.zeros((num_judges, num_evals, num_evals))

    with torch.no_grad():
        for i, j, k, n_j, n_k, r in test_loader:
            i, j, k = i.to(device), j.to(device), k.to(device)
            n_j, n_k, r = n_j.to(device), n_k.to(device), r.to(device)

            r = r.long()
            logits = model(i, j, k, n_j, n_k)
            loss = loss_fn(logits, r)
            per_sample_loss = F.cross_entropy(logits, r, reduction='none').cpu().numpy()

            total_test_loss += loss.item() * r.size(0)
            
            # Compute per-sample loss and accumulate in loss_matrix
            i_np = i.cpu().numpy()
            j_np = j.cpu().numpy()
            k_np = k.cpu().numpy()
            for n in range(r.size(0)):
                row = i_np[n]
                col_j = j_np[n]
                col_k = k_np[n]
                loss_matrix[row][col_j][col_k] += per_sample_loss[n]
                count_matrix[row][col_j][col_k] += 1

    avg_loss_matrix = np.divide(
        loss_matrix, count_matrix, 
        out=np.zeros_like(loss_matrix), 
        where=count_matrix != 0
    )

    avg_test_loss = total_test_loss / len(test_loader.dataset)


    log_path = path + 'log_train.txt'
    with open(log_path, 'w') as f:
        f.write(f'dataset = {path}\n')
        f.write(f'train_datasize = {len(train_comps)}\n')
        f.write(f'test_datasize = {len(test_comps)}\n')
        f.write(f'batch_size = {batch_size}\n\n')
        f.write(f'model = {model.__class__.__name__}\n')
        f.write(f'num_judges = {num_judges}\n')
        f.write(f'num_evals = {num_evals}\n')
        f.write(f'normalize = {NORMALIZE}\n')
        f.write(f'dim = {d}\n\n')
        f.write(f'lr = {lr}\n')
        f.write(f'weight_decay = {weight_decay}\n')
        f.write(f'epochs = {max_epochs}\n\n')
        f.write(f'Minimum Train Loss = {np.round(min(loss_history),4)}\n')
        f.write(f'Test Loss = {np.round(avg_test_loss,4)}\n\n')
        f.write(f'Test Loss Matrix:\n{np.round(avg_loss_matrix,4)}\n\n')
        f.write(f'Test Counts Matrix:\n{np.round(count_matrix,4)}')
    
    print(f"Training log written to {log_path}")

    if TRUST:
        print("Now computing eigentrust scores")
        T = compute_trust_matrix_ties(model, device)

        np.set_printoptions(formatter={'float': '{:8.4f}'.format})
        t = eigentrust(T, alpha=0)

        trust_path = path + 'eigentrust.txt'
        with open(trust_path, 'w') as f:
            f.write("Trust matrix:\n")
            f.write(str(T.cpu().numpy())+"\n")
            f.write("EigenTrust scores:\n")
            f.write(str(t.cpu().numpy())+"\n")
        
        print(f"Eigentrust scores written to {trust_path}")