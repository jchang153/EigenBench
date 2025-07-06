import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import re
import json
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

"""
Code for training Bradley-Terry models with binary comparison data, r in {1,2}
"""

class PairwiseDataset(Dataset):
    def __init__(self, comparisons):
        """
        comparisons: list of tuples (i, j, k, r)
          i: int index of the judge
          j, k: int indices of the two models compared
          r:     1 if j wins, -1 if k wins
        """
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i, j, k, r = self.data[idx]
        return torch.tensor(i, dtype=torch.long), \
               torch.tensor(j, dtype=torch.long), \
               torch.tensor(k, dtype=torch.long), \
               torch.tensor(r, dtype=torch.float32)

class VectorBT(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        # judge embeddings u_i and model embeddings v_j
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        # initialize
        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)

    def forward(self, i_idx, j_idx, k_idx):
        u_i = self.u(i_idx)   # shape: (batch, d)
        v_j = self.v(j_idx)   # shape: (batch, d)
        v_k = self.v(k_idx)   # shape: (batch, d)
        # compute utility differences
        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)
        return torch.sigmoid(score_j - score_k)
    
class VectorBT_norm(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        # judge embeddings u_i and model embeddings v_j
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        # initialize
        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)

    def forward(self, i_idx, j_idx, k_idx):
        u_i = self.u(i_idx)   # shape: (batch, d)
        v_j = self.v(j_idx)   # shape: (batch, d)
        v_k = self.v(k_idx)   # shape: (batch, d)
        # compute utility differences
        # latent strength: negative squared Euclidean distance
        score_j = -torch.sum((u_i - v_j) ** 2, dim=-1)
        score_k = -torch.sum((u_i - v_k) ** 2, dim=-1)
        return torch.sigmoid(score_j - score_k)
    
class VectorBT_bias(nn.Module):
    def __init__(self, num_models, d):
        super().__init__()
        # judge embeddings u_i and model embeddings v_j
        self.u = nn.Embedding(num_models, d)
        self.v = nn.Embedding(num_models, d)
        self.b = nn.Embedding(num_models, 1)
        # initialize
        nn.init.normal_(self.u.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.1)

    def forward(self, i_idx, j_idx, k_idx):
        u_i = self.u(i_idx)   # shape: (batch, d)
        v_j = self.v(j_idx)   # shape: (batch, d)
        v_k = self.v(k_idx)   # shape: (batch, d)
        b_i = self.b(i_idx)   # shape: (batch, 1)
        # compute utility differences
        score_j = torch.sum(u_i * v_j, dim=-1)
        score_k = torch.sum(u_i * v_k, dim=-1)
        return torch.sigmoid(score_j - score_k + b_i.squeeze(-1))

def train_vector_bt(model, dataloader, lr, weight_decay, max_epochs, device, save_path=None, normalize=False):
    """
    normalize adds the option of normalizing the 'v' vectors to be unit vectors between each optimization step.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    loss_history = []

    for epoch in range(1, max_epochs+1):
        total_loss = 0.0
        model.train()

        for i_idx, j_idx, k_idx, r in dataloader:
            i_idx = i_idx.to(device)
            j_idx = j_idx.to(device)
            k_idx = k_idx.to(device)
            r = r.to(device)

            r = 2-r # convert {1,2} to {1,0}

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
    plt.ylabel('Average BCE Loss')
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

def extract_comparisons(data, include_scenario=False):
    comparisons = []
    data_cleaned = []
    for i, item in enumerate(data):
        response = item['judge response']
        eval1_response = item['eval1 response']
        eval2_response = item['eval2 response']
        eval1_reflection = item['eval1 reflection']
        eval2_reflection = item['eval2 reflection']

        if response == None or eval1_response == None or eval2_response == None or eval1_reflection == None or eval2_reflection == None:
            continue

        e = re.search(r"Error in \w+ API call", response)
        e1 = re.search(r"Error in \w+ API call", eval1_response)
        e2 = re.search(r"Error in \w+ API call", eval2_response)
        e3 = re.search(r"Error in \w+ API call", eval1_reflection)
        e4 = re.search(r"Error in \w+ API call", eval2_reflection)
        if e or e1 or e2 or e3 or e4:
            print(f"Error in {i}th API call")
            continue

        m = re.search(r'<choice>(.)</choice>', response)
        if m:
            try:
                score = int(m.group(1))
                if include_scenario:
                    comparisons.append([item['scenario_index'], item['judge'], item['eval1'], item['eval2'], score])
                else:
                    comparisons.append([item['judge'], item['eval1'], item['eval2'], score])
                data_cleaned.append(item)
            except:
                print(f"No number found in the {i}th judge response")
                continue
        else:
            print(f"No <choice>…</choice> match found in the {i}th judge response")

    return comparisons, data_cleaned


if __name__ == "__main__":

    path = 'transcript/20250614_000000/'

    filepath = path + 'evaluations.json'
    cleaned_filepath = path + 'evaluations_cleaned.json'

    if not os.path.exists(cleaned_filepath):
        data = []
        with open(filepath, 'r') as file:
            data.extend(json.load(file))

        comparisons, data_cleaned = extract_comparisons(data)

        print("Loaded data has length", len(data))
        print("Formed", len(comparisons), "comparisons\n")

        with open(cleaned_filepath, "w") as file:
            json.dump(data_cleaned, file, indent=4)
            print(f"Cleaned transcript written to {cleaned_filepath}\n")

    else:
        data = []
        with open(cleaned_filepath, 'r') as file:
            data.extend(json.load(file))

        comparisons, data_cleaned = extract_comparisons(data)

        print("Loaded cleaned data has length", len(data))
        print("Formed", len(comparisons), "comparisons\n")

    # if data contains personas, see dimensionality.ipynb for mapping function

    TEST = True
    NORMALIZE = True
    batch_size = 32
    num_models = 5
    d = 4

    lr = 1e-3
    weight_decay = 0
    max_epochs = 100
    device = 'cpu'

    model = VectorBT(num_models, d)
    # model = VectorBT_norm(num_models, d)
    # model = VectorBT_bias(num_models, d)

    print("Ready to train with the following parameters:")
    print("model:", model.__class__.__name__)
    print("batch_size:", batch_size)
    print("num_models:", num_models)
    print("dim:", d)
    print("lr:", lr)
    print("weight_decay:", weight_decay)
    print("max_epochs:", max_epochs, "\n\n")


    if not TEST:
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
            normalize=NORMALIZE
        )

        log_path = path + 'log_train.txt'
        with open(log_path, 'w') as f:
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

    else:
        # create train-test-split
        from sklearn.model_selection import train_test_split
        train_comps, test_comps = train_test_split(
            comparisons,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        train_loader = DataLoader(PairwiseDataset(train_comps), batch_size=32, shuffle=True)
        test_loader  = DataLoader(PairwiseDataset(test_comps),  batch_size=32, shuffle=False)

        loss_history = train_vector_bt(
            model, 
            train_loader,
            lr=lr, 
            weight_decay=weight_decay,
            max_epochs=max_epochs, 
            device=device,
            save_path=path,
            normalize=NORMALIZE
        )

        print('Now evaluating the model on the test set\n')
        model.eval()
        loss_fn = nn.BCELoss()
        total_test_loss = 0.0

        with torch.no_grad():
            for i_idx, j_idx, k_idx, r in test_loader:
                i_idx, j_idx, k_idx = i_idx.to(device), j_idx.to(device), k_idx.to(device)
                r = r.to(device)
                p = model(i_idx, j_idx, k_idx)
                loss  = loss_fn(p, r)

                total_test_loss += loss.item() * r.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)


        log_path = path + 'log_train.txt'
        with open(log_path, 'w') as f:
            f.write(f'dataset = {path}\n')
            f.write(f'train_datasize = {len(train_comps)}\n')
            f.write(f'test_datasize = {len(test_comps)}\n')
            f.write(f'batch_size = {batch_size}\n\n')
            f.write(f'model = {model.__class__.__name__}\n')
            f.write(f'num_models = {num_models}\n')
            f.write(f'dim = {d}\n\n')
            f.write(f'lr = {lr}\n')
            f.write(f'weight_decay = {weight_decay}\n')
            f.write(f'epochs = {max_epochs}\n\n')
            f.write(f'Minimum Train Loss = {min(loss_history)}\n')
            f.write(f'Test Loss = {avg_test_loss}\n')
        
        print(f"Log written to {log_path}")