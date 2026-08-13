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

def solve(mat,bound):
    d = len(mat[0])-1
    k = len(mat)
    z3_s = z3.Solver()

    a = []
    for i_k in range(k):
        a.append([])
        for i_d in range(d):
            a_current = z3.Int(f"a_{i_k}_{i_d}")
            z3_s.add(a_current <= bound + mat[i_k][i_d])
            z3_s.add(a_current >= -1 * bound + mat[i_k][i_d])
            a[-1].append(a_current)
    b = []
    for i_k in range(k):
        b_current = z3.Int(f"b_{i_k}")
        z3_s.add(b_current <= bound + mat[i_k][-1])
        z3_s.add(b_current >= -1 * bound + mat[i_k][-1])
        b.append(b_current)
        z3_s.add(z3.Not(z3.And(z3.And(*[v == 0 for v in a[i_k]]), b[i_k] == 0)))
    int_mat = []
    hp1, hp2 = gen_hypercubes(d)

    edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()

    l = hp1.shape[1]

    for i_l in range(l):
        int_mat.append([])
        for i_k in range(k):
            edge_const = z3.Or(z3.And(sum([a[i_k][i_d] * hp1[i_d, i_l] for i_d in range(d)]) - b[i_k] > 0,
                                      sum([a[i_k][i_d] * hp2[i_d, i_l] for i_d in range(d)]) - b[i_k] < 0),
                               z3.And(sum([a[i_k][i_d] * hp1[i_d, i_l] for i_d in range(d)]) - b[i_k] < 0,
                                      sum([a[i_k][i_d] * hp2[i_d, i_l] for i_d in range(d)]) - b[i_k] > 0))
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
                i_k, i_d = int(i_k), int(i_d)
                print(var, val, i_k, i_d)
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


def solve_adjust_signs(mat,  bound):
    d = len(mat[0])-1
    k = len(mat)
    # method 0:
    # find thing where all coeffs are "bound" away from given
    # method 1:
    # adjust signs as well (different sub)

    #further methods if these fail.


    z3_s = z3.Solver()

    a = []
    a_coeff = []
    for i_k in range(k):
        a.append([])
        a_coeff.append([])
        for i_d in range(d):
            a_current = z3.Int(f"a_{i_k}_{i_d}")
            z3_s.add(a_current <= 1)
            z3_s.add(a_current >= -1)
            a[-1].append(a_current)
            a_coeff[-1].append(mat[i_k][i_d])
    b = []
    for i_k in range(k):
        b_current = z3.Int(f"b_{i_k}")
        z3_s.add(b_current <= bound + mat[i_k][-1])
        z3_s.add(b_current >= -1 * bound - mat[i_k][-1])
        b.append(b_current)
        z3_s.add(z3.Not(z3.And(z3.And(*[v == 0 for v in a[i_k]]), b[i_k] == 0)))
    int_mat = []
    hp1, hp2 = gen_hypercubes(d)

    edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()

    l = hp1.shape[1]

    for i_l in range(l):
        int_mat.append([])
        for i_k in range(k):
            edge_const = z3.Or(
                z3.And(sum([a[i_k][i_d] * a_coeff[i_k][i_d] * hp1[i_d, i_l] for i_d in range(d)]) - b[i_k] > 0,
                       sum([a[i_k][i_d] * a_coeff[i_k][i_d] * hp2[i_d, i_l] for i_d in range(d)]) - b[i_k] < 0),
                z3.And(sum([a[i_k][i_d] * a_coeff[i_k][i_d] * hp1[i_d, i_l] for i_d in range(d)]) - b[i_k] < 0,
                       sum([a[i_k][i_d] * a_coeff[i_k][i_d] * hp2[i_d, i_l] for i_d in range(d)]) - b[i_k] > 0))
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
                i_k, i_d = int(i_k), int(i_d)
                val *= a_coeff[i_k][i_d]
                print(var, val, i_k, i_d)
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
    parser = argparse.ArgumentParser(
        prog='z3solver',
        description='looking for solutions of hyperplane conjecture with z3',
        epilog='Text at the bottom of help')
    #parser.add_argument("filename")
    parser.add_argument('method',)

    args = parser.parse_args()
    method = args.method
    filename =""# args.filename
    if filename != "":
        mat = torch.load(filename).tolist()
    else:
        init_mat = [
            #  x1-x8: -12 each                 | x9   x10  x11  x12  x13  x14
            [-12, -12, -12, -12, -12, -12, -12, -12, -12, -4, 18, 4, 66, -21, .5],  # H1
            [-12, -12, -12, -12, -12, -12, -12, -12, -12, 3, -15, -15, -65, 41, .5],  # H2
            [-12, -12, -12, -12, -12, -12, -12, -12, -12, 29, 5, -71, 7, -5, .5],  # H3
            [-12, -12, -12, -12, -12, -12, -12, -12, -11, -70, -5, -25, 4, -4, .5],  # H4
            [-12, -12, -12, -12, -12, -12, -12, -12, -12, 3, -51, -5, 13, 41, .5],  # H5
            [-12, -12, -12, -12, -12, -12, -12, -12, -13, -4, 69, 1, -23, -21, .5],  # H6
            [-12, -12, -12, -12, -12, -12, -12, -12, -13, -5, 18, 0, 35, 53, .5],  # H7
            [-12, -12, -12, -12, -12, -12, -12, -12, -12, 5, -34, 7, 10, -57, .5],  # H8
            [-12, -12, -12, -12, -12, -12, -12, -12, -12, 71, 7, 27, 6, 4, .5],  # H9
            [-12, -12, -12, -12, -12, -12, -12, -12, -13, 1, -18, -5, -52, -34, .5],  # H10
            [-12, -12, -12, -12, -12, -12, -12, -12, -13, -30, -6, 76, -3, 3, .5],  # H11
        ]

        d_1 = len(init_mat[0])
        k = len(init_mat)
        mat = [[2 * init_mat[i_k][i_d] for i_d in range(d_1)] for i_k in range(k)]
    if method==1:
        solve_adjust_signs(mat, 10)
    else:
        solve(mat, 10)
