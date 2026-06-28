from collections import Counter
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def gen_hypercubes(d, no_loading=False):
    try:
        if no_loading:
            raise(Exception("No loading files"))
        data = np.load(f"matrices{d}.npz")
        hpoints1 = data['hpoints1']
        hpoints2 = data['hpoints2']
        return hpoints1, hpoints2
    except:
        #non reduced
        hpoints1 = np.ones((d,2**(d-1)*d))
        hpoints2 = np.ones((d,2**(d-1)*d))

        #testing - gen all pairs, gen half
        j=0
        for i1 in range(2**d):
            for i2 in range(d):

                if 1<<i2>i1:
                    break
                if (i1&(1<<i2))!=0:
                    v1=i1
                    v2=i1^(1<<i2)
                    for j2 in range(d):
                        hpoints1[j2, j] = (-2* ((v1&(1<<j2)) !=0 ))+1
                        hpoints2[j2, j] = (-2* ((v2&(1<<j2)) !=0 ))+1
                    j += 1
        np.savez(f"matrices{d}.npz", hpoints1=hpoints1, hpoints2=hpoints2)
        return hpoints1, hpoints2

def gen_reducehypercubes(d, b, no_loading=False):
    b = tuple(b)
    l_b = len(b)
    if sum(b)!=d:
        raise(Exception("Invalid reduction (sum of b incorrect)"))
    try:
        if no_loading:
            raise(Exception("No loading files"))
        data = np.load(f"matrices_red{d}_{b}.npz")
        hpoints1 = data['hpoints1']
        hpoints2 = data['hpoints2']
        p_counts = data['p_counts']
        return hpoints1, hpoints2, p_counts
    except Exception as E:
        # non reduced
        edge_count = Counter()
        #should look at the c program to see how reduction accomplished
        def reducepoint(p):
            b_ind = 0
            cur_b = b[0]
            out = [0] * l_b
            for i, c in enumerate(p):
                if i>cur_b-1:
                    b_ind += 1
                    cur_b += b[b_ind]
                out[b_ind] += c
            return tuple(out)
        for i1 in range(2 ** d):
            for i2 in range(d):
                v1 = i1
                v2 = i1 ^ (1 << i2)
                v1, v2 = min(v1, v2), max(v1, v2)
                p1 = reducepoint([(-2 * ((v1 & (1 << j2)) != 0)) + 1 for j2 in range(d)])
                p2 = reducepoint([(-2 * ((v2 & (1 << j2)) != 0)) + 1 for j2 in range(d)])
                edge_count[(p1,p2)]+=1
        l = len(edge_count)
        p_counts = [0] * l
        hpoints1 = np.ones((l_b,l))
        hpoints2 = np.ones((l_b,l))
        for i, P in enumerate(edge_count):
            p1,p2 = P
            p_counts[i] = edge_count[(p1,p2)]
            for i2 in range(l_b):
                hpoints1[i2, i] = p1[i2]
                hpoints2[i2, i] = p2[i2]
        p_counts = np.diag(p_counts)/2
        np.savez(f"matrices_red{d}_{b}.npz", hpoints1=hpoints1, hpoints2=hpoints2, p_counts=p_counts)
        return hpoints1, hpoints2, p_counts

def intersection_matrix(d, h, edges):
    A = h.reshape(-1, d + 1)[:, :-1]
    b = h.reshape(-1, d + 1)[:, -1]
    intersection_func = lambda e: torch.multiply(
        torch.matmul(A, e[0:d]) - b,
        torch.matmul(A, e[d:]) - b
    )
    return torch.vmap(intersection_func, in_dims=0, out_dims=0)(edges.reshape(-1,d*2))

def get_k(d,h):
    return h.reshape(-1,d+1).shape[0]

def count(d,h,edges):
    return torch.sum(torch.vmap(torch.max)(intersection_matrix(d,h.reshape(-1,d+1), edges.reshape(-1,d*2))<0))

def count_reduced(b,h,edges, p_counts):
    d=len(b)
    x= torch.vmap(torch.max)(intersection_matrix(d,h.reshape(-1,d+1), edges.reshape(-1,d*2))<0).double()
    x= torch.matmul(p_counts, x)
    return torch.sum(x)


def performance(d,c):
    return c/(d*(2**(d-1)))

def apply_G(d,h,G,G_agg, edge_subset):
    int_matrix = intersection_matrix(d, h, edge_subset)
    agg = G_agg(torch.vmap(G)(int_matrix))
    return agg

def run_adam(d, h_init, G, G_agg, batch_size, edge_set, total_iter=60, randomness=False):
    edge_loader = DataLoader(edge_set, batch_size=batch_size, shuffle=True)
    h = torch.tensor(h_init).float().requires_grad_(True)
    optimizer = torch.optim.Adam([h],lr=0.01)
    if batch_size>-1:
        for i in range(total_iter):
            for edge_subset in edge_loader:

                # iterate x timees, iterate through edges
                optimizer.zero_grad()
                agg=apply_G(d,h,G,G_agg, edge_subset)
                agg.backward()

                if randomness:
                    with torch.no_grad():
                        if h.grad is not None:
                            # Noise scale proportional to current gradient magnitude or fixed decay
                            noise = torch.randn_like(h.grad) * 0.001
                            h.grad += noise
                optimizer.step()
    else:
        for i in range(total_iter):
            # iterate x timees, iterate through edges
            optimizer.zero_grad()
            agg=apply_G(d,h,G,G_agg, edge_set)
            agg.backward()
            if randomness:
                with torch.no_grad():
                    if h.grad is not None:
                        # Noise scale proportional to current gradient magnitude or fixed decay
                        noise = torch.randn_like(h.grad) * 0.001
                        h.grad += noise
            optimizer.step()
    print(f"k: {get_k(d,h)} d: {d} count: {count(d,h,edge_set)}")
    return h1

g_outer = lambda x: torch.sigmoid(x) + .1 * torch.exp(-(x - 1) ** 2)
g_inner = lambda x: torch.sum(torch.sigmoid(-1000 * x))
    # now get final count
g_dict = {"G_test": (lambda col : g_outer(g_inner(col)), lambda agg:-1*torch.sum(agg) ),
          #"G_original":lambda col: torch.sum(torch.sum(torch.log(1+torch.relu( -1*col)))),
          "G_sigmoid": (lambda col: 1/(1+torch.exp(100*torch.min(col))), lambda agg:-1*torch.sum(agg)),
          "G_relu_sum": (lambda col: torch.sum(torch.relu(-1*col)), lambda agg:-1*torch.sum(agg)),
          "G_relu_prod": (lambda col: 1-torch.prod(torch.relu(col)), lambda agg:-1*torch.sum(agg)),
          "G_relu_prod_1000": (lambda col: 1-torch.prod(torch.relu(1000*col)), lambda agg:-1*torch.sum(agg) ),
          "G_sigmoid_gates": (lambda col: torch.sigmoid(torch.sum(-20*col)+10), lambda agg: torch.sigmoid(torch.sum(20*agg)-10) )}

def basic_gdict_runner(k,d, g_dict, randomness=False, matrix_save_folder=None, csv_file=None, save_id=None):
    saved_files = []
    hpoints1, hpoints2 = gen_hypercubes(d)
    edge_set = torch.tensor(np.concatenate((hpoints1, hpoints2)).T).float()
    # run adam on pairs of G_func, G_agg_func
    out_list = []
    if save_id is None:
        save_id = "default"
    for name, funcs in g_dict.items():
        h_init = torch.rand(k*(d+1))*10-5
        G,G_agg = funcs
        h_out = run_adam(d,h_init,G,G_agg,k, edge_set,60, randomness=randomness)
        c= count(d,h_out,edge_set).item()
        out_list.append({"k":k, "d":d, "g_name": name, "count": c, "performance": performance(d,c)})
        print(out_list[-1])
        if matrix_save_folder is not None:
            fname = matrix_save_folder+"_"+name+"_" + save_id + ".npy"
            np.save(fname, h_out.detach().numpy())
            saved_files.append(fname)
    if csv_file is not None:
        pd.DataFrame(out_list).to_csv(csv_file)
        saved_files.append(csv_file)
    return saved_files

basic_gdict_runner(5,6,g_dict,randomness=False)
print("-----------------------")
basic_gdict_runner(5,6,g_dict,randomness=True)



