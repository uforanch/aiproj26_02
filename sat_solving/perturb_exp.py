import z3
import torch
import numpy as np
from src import gen_hypercubes, count

h = torch.load("z3_solution_6_7_00.pt")
k,d = h.shape
d-=1
hp1, hp2 = gen_hypercubes(d)
l = hp1.shape[1]
max_cut = d*2**(d-1)
edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()
exp1 = False
if exp1:
    for i in range(2**k):
        v=torch.tensor([.01*(-1 if (i>>j) % 2 == 0 else 1) for j in range(k)]).reshape(k).double()
        h[:,d]+=v
        c=count(d,h,edge_set)
        if c==max_cut:
            print("perturb found")
            torch.save(h, "pertrun_solution_6_7.pt")
        h[:,d]-=v
hp1 = torch.tensor(hp1).double()
hp2 = torch.tensor(hp2).double()
A = h.reshape(-1, d + 1)[:, :-1]
b = h.reshape(-1, d + 1)[:, -1].reshape((-1,1))
v1 = torch.matmul(A, hp1)-b
v2 = torch.matmul(A, hp2)-b
one_zero = 0
two_zero = 0
for i_l in range(l):
    for i_k in range(k):
        if (v1[i_k, i_l] == 0 and v2[i_k,i_l] != 0) or (v1[i_k,i_l] != 0 and v2[i_k,i_l] == 0) :
            print(i_k, h[i_k,:])
            print(hp1[:,i_l], hp2[:,i_l])
            print(v1[i_k, i_l], v2[i_k, i_l])
            print("----")
            one_zero+=1
        elif  (v1[i_k, i_l] == 0 and v2[i_k,i_l] == 0):
            two_zero+=1

print(f"final counts {one_zero} vs {two_zero}")