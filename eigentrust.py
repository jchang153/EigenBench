import torch
import torch.nn.functional as F

from BT import VectorBT, VectorBT_norm, VectorBT_bias
import numpy as np
from tqdm import tqdm

def load_vector_bt(path, num_models, d, device='cpu'):
    model = VectorBT(num_models, d).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def compute_trust_matrix(model, device='cpu'):
    U = model.u.weight.data.to(device)  # shape (N, d)
    V = model.v.weight.data.to(device)  # shape (N, d)

    # Compute raw scores S_ij = u_i · v_j
    S = U @ V.t()                       # shape (N, N)
    S = torch.exp(S)
    return S

def row_normalize(S):
    # Avoid rows summing to zero
    row_sums = S.sum(dim=1, keepdim=True)         # (N,1)
    C = S / row_sums
    return C

def damp_matrix(C, alpha=0.15):
    M = C.size(0)
    E = torch.full_like(C, 1.0 / M)
    return (1 - alpha) * C + alpha * E

def eigentrust(C, alpha=0.15, tol=1e-6, max_iter=1000):
    T = damp_matrix(C, alpha)

    # Initialize trust vector
    t = torch.full((T.size(0),), 1.0 / T.size(0), device=T.device)

    print("Starting eigentrust iteration")
    for _ in tqdm(range(max_iter)):
        t_next = t @ T
        if torch.norm(t_next - t, p=1) < tol:
            break
        t = t_next

    return t_next

if __name__ == "__main__":

    path = 'transcript/20250614_000000/'
    
    model_path   = path + 'model.pt'
    num_models   = 5
    d            = 6
    device       = 'mps'

    model = load_vector_bt(model_path, num_models, d, device)

    S = compute_trust_matrix(model, device)
    C = row_normalize(S)

    np.set_printoptions(formatter={'float': '{:8.4f}'.format})
    print("Trust matrix:")
    print(S.cpu().numpy())
    print("Row-normalized:")
    print(C.cpu().numpy() )

    t = eigentrust(C, alpha=0)

    log_path = f'{path}eigentrust.txt'
    with open(log_path, 'w') as f:
        f.write("Trust matrix:\n")
        f.write(str(S.cpu().numpy())+"\n")
        f.write("Row-normalized:\n")
        f.write(str(C.cpu().numpy())+"\n")
        f.write("EigenTrust scores:\n")
        f.write(str(t.cpu().numpy())+"\n")
