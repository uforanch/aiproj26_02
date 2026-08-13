"""
process:

pick unintersected column
find in that column a zero-nonzero pair or zero zero pair
find an adjustment that doesn't make anything else unintersected

tracking:
track which columns are intersections and which aren't, and how many and what planes intersect
bring up if the plane is the only intersection for multiple columns

adjustment: 0-0 - increase/descrease in sign difference index
0- nonzero - small adjustment in opposing vector  to non-sign difference indices

choosing and adjusting:
find unintersected column
look for zero-nonzero pair or zero zero pair
find an adjustment that doesn't make anything else unintersected
----

more detail:
matrix for which columns have one, zero, or multiple intersections.
for each plane, list which edges it's the only intersection for


probably limit to 500 random adjustments, print out when it makes forward or backward progres,
debug to see things happening
"""
import torch
import numpy as np
import random
from src import gen_hypercubes

h = torch.load("z3_solution_6_7_00.pt")
k,d = h.shape
d-=1
hp1, hp2 = gen_hypercubes(d)
l = hp1.shape[1]
max_cut = d*2**(d-1)
edge_set = torch.tensor(np.concatenate((hp1, hp2)).T).double()

hp1 = torch.tensor(hp1).double()
hp2 = torch.tensor(hp2).double()

diff_mask = hp1!=hp2

random_plane = None

k_list = list(range(k))


for i in range(500):
    A = h.reshape(-1, d + 1)[:, :-1]
    b = h.reshape(-1, d + 1)[:, -1].reshape((-1, 1))
    v1 = torch.matmul(A, hp1) - b
    v2 = torch.matmul(A, hp2) - b
    if random_plane is not None:
        v1_val = v1[random_plane, random_uncut_edge]
        v2_val = v2[random_plane, random_uncut_edge]
        print("new values ", v1_val, v2_val)
        if not((v1_val>0 and v2_val<0) or (v1_val<0 and v2_val>0)):
            print("process failed - bad adjustment")
            break
    print("----")
    #intersections
    int_mat = torch.multiply(v1, v2)
    x=torch.sum(int_mat>0, dim=0)
    bad_cases = torch.sum(torch.sum(int_mat>0, dim=0)==k)
    if bad_cases>0:
        print("process failed - failed edge")
        #break

    int_totals = torch.sum(int_mat<0, dim=0)
    print("edges_cut ", torch.sum(int_totals>0).item())
    uncut_edges = torch.nonzero(int_totals==0)
    l2=len(uncut_edges.reshape(-1))
    random_uncut_edge = torch.randint(0,len(uncut_edges.reshape(-1)),(1,)).item()
    v1_vec = v1[:,random_uncut_edge]
    v2_vec = v2[:,random_uncut_edge]

    # while True:
    #     random_plane = torch.randint(0,k,(1,)).item()
    #
    #     v1_val = v1[random_plane, random_uncut_edge]
    #     v2_val = v2[random_plane, random_uncut_edge]
    #     if v1_val==0 or v2_val==0:
    #         break
    # selectable_edges = torch.nonzero(
    #     torch.logical_or(torch.abs(v1_vec) == torch.min(torch.abs(v1_vec)), torch.abs(v2_vec) == torch.min(torch.abs(v2_vec)))).reshape(
    #     -1)
    # random_choice = torch.randint(0,len(selectable_edges),(1,))
    min_abs = 10**6
    random_plane = None
    random.shuffle(k_list)
    for i_k in k_list:
        v1_temp_val = v1_vec[i_k]
        v2_temp_val = v2_vec[i_k]
        if ((v1_temp_val> 0 and v2_temp_val< 0) or
                (v1_temp_val< 0 and v2_temp_val> 0)):
            continue
        if abs(v1_temp_val) < min_abs or abs(v2_temp_val) < min_abs:
            random_plane = i_k
            min_abs = abs(v1_temp_val)

    v1_val = v1[random_plane, random_uncut_edge]
    v2_val = v2[random_plane, random_uncut_edge]
    print("old values ", v1_val, v2_val)

    #doing this nonvectorized... k is small
    #don't have time for this
    if v1_val==0 and v2_val==0:
        for i_d in range(d):
            if diff_mask[i_d, random_uncut_edge]==True:
                h[random_plane, i_d]+=.01
    elif v1_val==0 and v2_val!=0:
        s = -1 if v2_val>0 else 1
        for i_d in range(d):
            if diff_mask[i_d, random_uncut_edge]==False:
                h[random_plane, i_d]+= s*.01/d * hp1[i_d, random_uncut_edge]
    elif v2_val==0 and v1_val!=0:
        s = -1 if v1_val>0 else 1
        for i_d in range(d):
            if diff_mask[i_d, random_uncut_edge]==False:
                h[random_plane, i_d]+= s*.01/d * hp2[i_d, random_uncut_edge]
    elif abs(v1_val)<abs(v2_val):
        s = -1  if v1_val>0 else 1
        for i_d in range(d):
            if diff_mask[i_d, random_uncut_edge]==True:
                h[random_plane, i_d]+= s*(abs(v1_val)+.01) * hp1[i_d, random_uncut_edge]
    elif abs(v2_val)<=abs(v1_val):
        s = -1  if v2_val>0 else 1
        for i_d in range(d):
            if diff_mask[i_d, random_uncut_edge]==True:
                h[random_plane, i_d]+= s*(abs(v2_val)+.01) * hp2[i_d, random_uncut_edge]



