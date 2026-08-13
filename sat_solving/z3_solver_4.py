import argparse
import z3
import torch
import numpy as np
import random
from src import gen_hypercubes, count

"""
goal: load config, find something near it or don't 
just print output given how this works

"""

def solve(k,d, mat,bound):
    d_m = len(mat[0])-1
    k_m = len(mat)
    z3_s = z3.Solver()

    a = dict()
    for i_k in range(k):
        for i_d in range(d):
            if i_k >= k_m and i_d <d_m:
                a_current = z3.Int(f"a_{i_k}_{i_d}")
                z3_s.add(a_current <= bound )
                z3_s.add(a_current >= -1 * bound )
                a[f"a_{i_k}_{i_d}"] = a_current
            elif i_d==d_m:
                a_current = z3.Int(f"a_s_{i_k}")
                z3_s.add(a_current <= bound )
                z3_s.add(a_current >= -1 * bound )
                a[f"a_s_{i_k}"]=a_current


    b = []
    for i_k in range(k):
        b_current = z3.Int(f"b_{i_k}")
        z3_s.add(b_current <= bound )
        z3_s.add(b_current >= -1 * bound )
        b.append(b_current)
        if i_k>=k_m:
            z3_s.add(z3.Not(z3.And(z3.And(*[a[f"a_{i_k}_{i_d}"] ==0 for i_d in range(d_m)]), a[f"a_s_{i_k}"]==0, b[i_k] == 0)))
        else:
            z3_s.add(z3.Not(
                z3.And( a[f"a_s_{i_k}"] == 0, b[i_k] == 0)))
    int_mat = []
    hp1, hp2 = gen_hypercubes(d)

    edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()

    l = hp1.shape[1]
    def eval_cond(h, i_k):
        if i_k<k_m:
            return sum([mat[i_k][i_d] * h[i_d] if i_d <d_m else a[f"a_s_{i_k}"] * h[i_d] for i_d in range(d)]) - b[i_k]
        else:
            return sum([a[f"a_{i_k}_{i_d}"] * h[i_d] if i_d <d_m else a[f"a_s_{i_k}"] * h[i_d] for i_d in range(d)]) - b[i_k]
    for i_l in range(l):
        int_mat.append([])
        for i_k in range(k):
            edge_const = z3.Or(z3.And(eval_cond(hp1[:,i_l], i_k) > 0,
                                      eval_cond(hp2[:,i_l], i_k)< 0),
                               z3.And(eval_cond(hp1[:,i_l], i_k)< 0,
                                      eval_cond(hp2[:,i_l], i_k) > 0))
            int_mat[-1].append(edge_const)
    z3_s.add(z3.And(*[z3.Or(*[int_mat[i_l][i_k] for i_k in range(k)]) for i_l in range(l)]))

    h = torch.zeros([k, d + 1]).double()

    print(z3_s.check())
    if z3_s.check() == z3.sat:
        print("solving")
        model = z3_s.model()
        for decl in model.decls():
            var = decl.name()
            val = model[decl].as_long()
            if var.startswith("a"):
                _, i_k, i_d = var.split("_")
                if i_k != "s":
                    i_k, i_d = int(i_k), int(i_d)
                    print(var, val, i_k, i_d)
                    h[i_k, i_d] = val
                else:
                    i_k = int(i_d)
                    for i_d in range(d_m,d):
                        h[i_k, i_d] = val

            elif var.startswith("b"):
                _, i_k = var.split("_")
                i_k = int(i_k)
                print(var, val, i_k)
                h[i_k, d] = val
        print("---")
        print(h)
        torch.save(h, f"z3_solution_{k}_{d}.pt")
        print("---")
        print(count(d, h, edge_set))
    elif z3_s.check() == z3.unsat:
        print("Why it failed:", z3_s.unsat_core())


if __name__  == "__main__":
    #parser = argparse.ArgumentParser(
    #    prog='z3solver',
    #    description='looking for solutions of hyperplane conjecture with z3',
    #    epilog='Text at the bottom of help')
    #parser.add_argument("filename")
    #parser.add_argument('method',)

    #args = parser.parse_args()
    #method = args.method
    filename =""# args.filename
    if filename != "":
        mat = torch.load(filename).tolist()
    else:
        init_mat = [
            [-2,-2,-2,-2,-2,-2,1,3,-8,-1,.5],
            [-2, -2, -2, -2, -2, -2, -1, -3, 8, 1, .5],
            [-2, -2, -2, -2, -2, -2, -1, 8, 3, -1, .5],
            [-2, -2, -2, -2, -2, -2, 1, -8, -3, 1, .5],
            [-2, -2, -2, -2, -2, -2, 4,-1,1,-7, .5],
            [-2, -2, -2, -2, -2, -2, -4, 1, -1, 7, .5],
            [-2, -2, -2, -2, -2, -2, -7,-1,1,-4, .5],
            [-2, -2, -2, -2, -2, -2, 7, 1, -1, 4, .5],
        ]

        d_1 = len(init_mat[0])
        k = len(init_mat)
        mat = [[2 * init_mat[i_k][i_d] for i_d in range(d_1)] for i_k in range(k)]
        solve(11,14,mat, 15)