import torch
import numpy as np
import random
from src import gen_hypercubes,count

h = torch.load("z3_solution_6_7_00.pt")
k,d = h.shape
d-=1
hp1, hp2 = gen_hypercubes(d)
l = hp1.shape[1]
max_cut = d*2**(d-1)
edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()

hp1 = torch.tensor(hp1).double()
hp2 = torch.tensor(hp2).double()
c=count(d,h,edge_set)

for i in range(1000):
    v=torch.randn_like(h) * .02 -.01
    c2 = count(d, h, edge_set)
    if c2 == max_cut:
        h2 = h+v
        torch.save(h2,"perturb_exp_solution_6_7_00.pt")
        print("solved")
        break
    if c2>c:
        c=c2
        print("new high score ", c2)


