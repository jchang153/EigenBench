import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import re
import json
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt

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


def train_vector_bt(model, dataloader, lr, weight_decay, max_epochs, device, save_path=None):#, tol=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    loss_history = []

    # prev_u = model.u.weight.data.clone()
    # prev_v = model.v.weight.data.clone()

    for epoch in range(1, max_epochs+1):
        total_loss = 0.0
        model.train()

        for i_idx, j_idx, k_idx, r in dataloader:
            i_idx = i_idx.to(device)
            j_idx = j_idx.to(device)
            k_idx = k_idx.to(device)
            r = r.to(device)

            # r = (r+1.0)/2.0 # convert {1,-1} to {1,0}
            r = 2-r # convert {1,2} to {1,0}

            p = model(i_idx, j_idx, k_idx)
            loss = loss_fn(p, r) # cross entropy loss with classifying 0/1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * r.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)

        # delta_u = torch.max(torch.abs(model.u.weight.data - prev_u)).item()
        # delta_v = torch.max(torch.abs(model.v.weight.data - prev_v)).item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3d}, Loss = {avg_loss:.4f}")
            # print(delta_u, delta_v)

        # # check convergence
        # if epoch > 50 and delta_u < tol and delta_v < tol:
        #     print(f"Converged at epoch {epoch}: Δu_max={delta_u:.2e}, Δv_max={delta_v:.2e} < tol={tol:.2e}")
        #     break

        # # cache current weights for next comparison
        # prev_u.copy_(model.u.weight.data)
        # prev_v.copy_(model.v.weight.data)

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

        # log_path = save_path+'log_train.txt'
        # with open(log_path, 'a') as f:
        #     f.write(f'Finished training after {len(loss_history)} epochs\n')
        #     f.write(f'Last difference in u and v embeddings was \ndelta_u: {delta_u}\ndelta_v: {delta_v}')
        # print(f"Log saved to {log_path}")

    return loss_history

def extract_comparisons(data):
    comparisons = []
    data_cleaned = []
    for i, item in enumerate(data):
        response = item['judge response']
        eval1_response = item['eval1 response']
        eval2_response = item['eval2 response']
        # test with omitting self judgments
        # if item['judge'] == item['eval1'] or item['judge'] == item['eval2']:
        #     continue
        # test with omitting judgments where j <= k and j >= k
        # if item['eval1'] >= item['eval2']:
        #     continue
        # test with omitting jugments for model j
        # if item['eval1'] == 1 or item['eval2'] == 1:
        #     continue
        # test with omitting model idx judging model idx+1
        # if item['judge'] == 4 and (item['eval1'] == 0 or item['eval1'] == 0):
        #     continue
        # test with omitting model idx as judge or candidate
        # if item['judge'] == 1 or item['eval1'] == 1 or item['eval2'] == 1:
        #     continue

        if response == None or eval1_response == None or eval2_response == None:
            continue

        e = re.search(r"Error in \w+ API call", response)
        e1 = re.search(r"Error in \w+ API call", eval1_response)
        e2 = re.search(r"Error in \w+ API call", eval2_response)
        if e or e1 or e2:
            print(f"Error in {i}th API call")
            continue

        m = re.search(r'<choice>(.)</choice>', response)
        if m:
            try:
                score = int(m.group(1))

                comparisons.append([item['judge'], item['eval1'], item['eval2'], score])
                data_cleaned.append(item)
            except:
                print(f"No number found in the {i}th judge response")
                continue
        else:
            print(f"No <choice>…</choice> match found in the {i}th judge response")

    return comparisons, data_cleaned


if __name__ == "__main__":

    path = 'transcript/20250609_180000/'

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

    # func = {'(0, 0)': 0,
    #     '(0, 1)': 1,
    #     '(0, 2)': 2,
    #     '(0, 3)': 3,
    #     '(0, 4)': 4,
    #     '(1, 0)': 5,
    #     '(1, 1)': 6,
    #     '(1, 2)': 7,
    #     '(1, 3)': 8,
    #     '(1, 4)': 9,
    #     '(2, 0)': 10,
    #     '(2, 1)': 11,
    #     '(2, 2)': 12,
    #     '(2, 3)': 13,
    #     '(2, 4)': 14,
    #     '(3, 0)': 15,
    #     '(3, 1)': 16,
    #     '(3, 2)': 17,
    #     '(3, 3)': 18,
    #     '(3, 4)': 19,
    #     '(4, 0)': 20,
    #     '(4, 1)': 21,
    #     '(4, 2)': 22,
    #     '(4, 3)': 23,
    #     '(4, 4)': 24,}
    
    # mapped_comparisons = []
    # for item in comparisons:
    #     l = [func[x] for x in item[:3]]
    #     l.append(item[3])
    #     mapped_comparisons.append(l)


    batch_size=32
    dataset = PairwiseDataset(comparisons)
    # dataset = PairwiseDataset(mapped_comparisons)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_models = 5
    d = 6

    lr = 1e-3
    weight_decay = 0
    max_epochs = 200

    print("Ready to train with the following parameters:")
    print("batch_size:", batch_size)
    print("num_models:", num_models)
    print("dim:", d)
    print("lr:", lr)
    print("weight_decay:", weight_decay)
    print("max_epochs:", max_epochs, "\n\n")

    model = VectorBT(num_models, d)
    # model = VectorBT_norm(num_models, d)
    loss_history = train_vector_bt(
        model, 
        dataloader,
        lr=lr, 
        weight_decay=weight_decay,
        max_epochs=max_epochs, 
        device='mps',
        save_path=path
    )

    log_path = path + 'log_train.txt'
    with open(log_path, 'w') as f:
        f.write(f'dataset = {path}\n')
        f.write(f'datasize = {len(comparisons)}\n')
        f.write(f'num_models = {num_models}\n')
        f.write(f'dim = {d}\n')
        f.write(f'lr = {lr}\n')
        f.write(f'weight_decay = {weight_decay}\n')
        f.write(f'epochs = {max_epochs}\n\n')
        f.write(f'Minimum Loss = {min(loss_history)}')