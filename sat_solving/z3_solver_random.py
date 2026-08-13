import z3
import torch
import numpy as np
from src import gen_hypercubes, count

d=6
k=1
bound = 30

z3_s= z3.Solver()

a=[]
for i_k in range(k):
    a.append([])
    for i_d in range(d):
        a_current = z3.Int(f"a_{i_k}_{i_d}")
        z3_s.add(a_current <= bound)
        z3_s.add(a_current >= -1*bound)
        a[-1].append(a_current)
b=[]
for i_k in range(k):
    b_current = z3.Int(f"b_{i_k}")
    z3_s.add(b_current <= bound)
    z3_s.add(b_current >= -1*bound)
    b.append(b_current)
    z3_s.add(z3.Not(z3.And(z3.And(*[v==0 for v in a[i_k]]), b[i_k]==0)))
int_mat = []
hp1, hp2 = gen_hypercubes(d)


edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()

num_samples = 10

selected_indices = torch.randperm(edge_set.shape[0])[:num_samples]

edge_est = edge_set[selected_indices]


l = edge_est.shape[0]

for i_l in range(l):
    int_mat.append([])
    for i_k in range(k):
        v1 = edge_set[i_l,0:d]
        v2 = edge_set[i_l,d:]
        edge_const = z3.Or(z3.And(sum([a[i_k][i_d] * v1[i_d].item() for i_d in range(d)])- b[i_k]>0,
                                  sum([a[i_k][i_d] * v2[i_d].item() for i_d in range(d)]) - b[i_k] <0),
                            z3.And(sum([a[i_k][i_d] * v1[i_d].item() for i_d in range(d)])- b[i_k] < 0,
                                sum([a[i_k][i_d] * v2[i_d].item() for i_d in range(d)])- b[i_k] > 0))
        int_mat[-1].append(edge_const)
z3_s.add(z3.And(*[z3.Or(*[int_mat[i_l][i_k] for i_k in range(k)]) for i_l in range(l)]))

h = torch.zeros([k, d+1]).double()

print(z3_s.check())
if z3_s.check() == z3.sat:
    print("solving")
    model = z3_s.model()
    for decl in model.decls():
        var = decl.name()
        val = model[decl].as_long()
        if var.startswith("a"):
            _, i_k, i_d = var.split("_")
            i_k,i_d=int(i_k),int(i_d)
            print(var, val, i_k, i_d)
            h[i_k,i_d] = val
        elif var.startswith("b"):
            _, i_k = var.split("_")
            i_k=int(i_k)
            print(var, val, i_k)
            h[i_k,d] = val
    print("---")
    print(h)
    torch.save(h, f"z3_solution_{k}_{d}.pt")
    print("---")
    print(count(d,h,edge_set))
elif z3_s.check() == z3.unsat:
    print("Why it failed:", z3_s.unsat_core())

#debug remove later
debug = False
if debug:
    print("--- solver check ---")
    print(z3_s.to_smt2())

    print("----- manual check ----")
    eval = lambda c : z3.is_true(model.evaluate(c))
    for i_l in range(l):
        for i_k in range(k):
            v1 = edge_set[i_l, 0:d]
            v2 = edge_set[i_l, d:]
            p1=sum([h[i_k,i_d] * v1[i_d].item() for i_d in range(d)])- h[i_k,d]
            p2=sum([h[i_k,i_d] * v2[i_d].item() for i_d in range(d)])- h[i_k,d]
            c1 = sum([a[i_k][i_d] * v1[i_d].item() for i_d in range(d)])- b[i_k]>=0
            c2 = sum([a[i_k][i_d] * v2[i_d].item() for i_d in range(d)]) - b[i_k] <= 0
            c3 = sum([a[i_k][i_d] * v1[i_d].item() for i_d in range(d)]) - b[i_k] <= 0
            c4 = sum([a[i_k][i_d] * v2[i_d].item() for i_d in range(d)]) - b[i_k] >= 0
            print(f"plane {i_k} edge {i_l}: \n ",
                  f"edge 1 : {hp1[:,i_l] }",
                  f"edge 2 : {hp2[:, i_l]}",
                  f"np evals: {p1}, {p2}\n",
                  f"model evals: {model.eval(sum([a[i_k][i_d] * hp1[i_d, i_l] for i_d in range(d)])- b[i_k])} {model.eval(sum([a[i_k][i_d] * hp2[i_d, i_l] for i_d in range(d)])- b[i_k])} \n"
                  f"constraints: {eval(c1)} and {eval(c2)} and {eval(c3)} and {eval(c4)}\n"
                  f"logic in int mat: {eval(int_mat[i_l][i_k])}")