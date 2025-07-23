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

from eigentrust import eigentrust, row_normalize, compute_trust_matrix

"""
Code for training Bradley-Terry models with binary comparison data, r in {1,2} -> {1,0}, with BCE loss

Or BTD model where inconsistent responses with order bias are marked as ties.
So we use a categorical CE loss with classes {tie,1,2} -> {0,1,2} 
"""

class PairwiseDataset(Dataset):
    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c, i, j, k, r = self.data[idx]
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

    def forward(self, c_idx, i_idx, j_idx, k_idx):
        judge_idx = c_idx * self.num_models + i_idx
        u_i_c = self.u(judge_idx)   # shape: (batch, d)
        v_j = self.v(j_idx)     # shape: (batch, d)
        v_k = self.v(k_idx)     # shape: (batch, d)

        score_j = torch.sum(u_i_c * v_j, dim=-1)
        score_k = torch.sum(u_i_c * v_k, dim=-1)

        log_lambda_i = self.log_lambda(judge_idx).squeeze(-1)
        tie_logit = log_lambda_i + 0.5 * (score_j + score_k)

        logits = torch.stack([tie_logit, score_j, score_k], dim=1)
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

        for c_idx, i_idx, j_idx, k_idx, r in dataloader:
            c_idx = c_idx.to(device)
            i_idx = i_idx.to(device)
            j_idx = j_idx.to(device)
            k_idx = k_idx.to(device)
            r = r.to(device)

            if use_btd:
                r = r.long()  # CrossEntropyLoss expects long tensor
                logits = model(c_idx, i_idx, j_idx, k_idx)
                loss = loss_fn(logits, r) # CE expects logits, unnormalized, as it has built in softmax
            else:
                p = model(i_idx, j_idx, k_idx)
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

    SEPARATE_CRITERIA = False

    path = 'transcript/20250722_000000/'

    # these comparisons were generated from test.ipynb, see "modifying extract comparisons..."
    comparisons = []
    with open(path + 'comparisons_with_ties.json', 'r') as file:
        comparisons.extend(json.load(file))

    if SEPARATE_CRITERIA:
        comparisons = [[i[:1][0]-1] + i[2:] for i in comparisons] # remove scenario index; convert criteria numbers to indices
        num_models = len(set([i[1] for i in comparisons]))
        num_criteria = len(set([i[0] for i in comparisons]))
    else:
        comparisons = [[0] + i[2:] for i in comparisons]  # remove scenario and criteria indices; make every datapoint associated with criterion 0
        num_models = len(set([i[1] for i in comparisons]))
        num_criteria = 1
    print("Formed", len(comparisons), "comparisons after accounting for ties")

    batch_size = 32
    d = 2

    lr = 1e-3
    weight_decay = 0
    max_epochs = 200
    device = 'cpu'


    model = VectorBTD(num_criteria, num_models, d)

    print(f"Ready to train {'BTD' if USE_BTD else 'BT'} model:")




    if TEST:
        train_comps, test_comps = train_test_split(
            comparisons,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        train_loader = DataLoader(PairwiseDataset(train_comps), batch_size=32, shuffle=True)
        test_loader = DataLoader(PairwiseDataset(test_comps), batch_size=32, shuffle=False)

        loss_history = train_vector_bt(
            model, 
            train_loader,
            lr=lr, 
            weight_decay=weight_decay,
            max_epochs=max_epochs, 
            device=device,
            save_path=path,
            normalize=NORMALIZE,
            use_btd=USE_BTD
        )

        print('Now evaluating the model on the test set\n')
        model.eval()
        
        if USE_BTD:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCELoss()
            
        total_test_loss = 0.0
        loss_matrix = np.zeros((num_criteria*num_models, num_models, num_models))
        count_matrix = np.zeros((num_criteria*num_models, num_models, num_models))

        with torch.no_grad():
            for c_idx, i_idx, j_idx, k_idx, r in test_loader:
                c_idx, i_idx, j_idx, k_idx = c_idx.to(device), i_idx.to(device), j_idx.to(device), k_idx.to(device)
                r = r.to(device)

                if USE_BTD:
                    r = r.long()
                    logits = model(c_idx, i_idx, j_idx, k_idx)
                    loss = loss_fn(logits, r)
                    per_sample_loss = F.cross_entropy(logits, r, reduction='none').cpu().numpy()
                else:
                    p = model(i_idx, j_idx, k_idx)
                    loss = loss_fn(p, r)
                    per_sample_loss = F.binary_cross_entropy(p, r, reduction='none').cpu().numpy()

                total_test_loss += loss.item() * r.size(0)
                
                # Compute per-sample loss and accumulate in loss_matrix
                c_idx_np = c_idx.cpu().numpy()
                i_idx_np = i_idx.cpu().numpy()
                j_idx_np = j_idx.cpu().numpy()
                k_idx_np = k_idx.cpu().numpy()
                for n in range(r.size(0)):
                    row = c_idx_np[n] * num_models + i_idx_np[n]
                    col_j = j_idx_np[n]
                    col_k = k_idx_np[n]
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
            f.write(f'num_models = {num_models}\n')
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
    
    else:
        dataset = PairwiseDataset(comparisons)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history = train_vector_bt(
            model, 
            dataloader,
            lr=lr, 
            weight_decay=weight_decay,
            max_epochs=max_epochs, 
            device=device,
            save_path=path,
            normalize=NORMALIZE,
            use_btd=USE_BTD
        )

        log_path = path + 'log_train.txt'
        with open(log_path, 'w') as f:
            f.write(f'model_type = {"BTD" if USE_BTD else "BT"}\n')
            f.write(f'dataset = {path}\n')
            f.write(f'datasize = {len(comparisons)}\n')
            f.write(f'batch_size = {batch_size}\n\n')
            f.write(f'model = {model.__class__.__name__}\n')
            f.write(f'num_models = {num_models}\n')
            f.write(f'dim = {d}\n\n')
            f.write(f'lr = {lr}\n')
            f.write(f'weight_decay = {weight_decay}\n')
            f.write(f'epochs = {max_epochs}\n\n')
            f.write(f'Minimum Loss = {min(loss_history)}')

    if TRUST:
        print("Now computing eigentrust scores")
        S = compute_trust_matrix(model, device)
        C = row_normalize(S)

        np.set_printoptions(formatter={'float': '{:8.4f}'.format})
        t = eigentrust(C, alpha=0)

        trust_path = path + 'eigentrust.txt'
        with open(trust_path, 'w') as f:
            f.write("Trust matrix:\n")
            f.write(str(S.cpu().numpy())+"\n")
            f.write("Row-normalized:\n")
            f.write(str(C.cpu().numpy())+"\n")
            f.write("EigenTrust scores:\n")
            f.write(str(t.cpu().numpy())+"\n")
        
        print(f"Eigentrust scores written to {trust_path}")