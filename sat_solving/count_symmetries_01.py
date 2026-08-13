"""
counting taking too long

absurd number of permuations with sign changes. we only want the ones that take the set to itself
there's also an absurd number of these that just take one vertex to the other

but we're doing small cases only

here's what we'll do
for each perumation - find sign changes that take one point to any other point
see if vertex set is perserved

to do this need:
obtain vertex set
apply permutation to all in vertex list
make new transformation that takes one to the other

"""
import itertools


def get_vertex_set_from_edges(edge_set):
    s=set()
    for edge in edge_set:
        s.add(tuple(edge[0]))
        s.add(tuple(edge[1]))
    return s

def apply_permutation(vertex_set, p):
    d=len(p)
    return [[v[p[i]] for i in range(d)] for v in vertex_set]

def get_transform_v1_v2(v1, v2):
    diff_list = [v1[i] * v2[i] for i in range(len(v1))]
    return lambda v: [v[i] * diff_list[i] for i in range(len(v))]


def get_if_original_set(new_lists, old_set):
    for l in new_lists:
        if tuple(l) not in old_set:
            return False
    return True

def count_symmetries(vertex_set):
    sym = 0
    d = len(vertex_set[0])
    vertex_set = list(set([tuple(v) for v in vertex_set]))
    for p in itertools.permutations(range(d)):
        list_00 = apply_permutation(vertex_set, p)
        v1 = list_00[0]
        for v in vertex_set:
            sign_map = get_transform_v1_v2(v1, v)
            list_01 = list(map(sign_map, list_00))
            sym+=get_if_original_set(list_01, vertex_set)
    return sym

#print(count_symmetries([[1,1,1]]))

print(count_symmetries([[1,1,1],[-1,1,1], [1,-1,1], [-1,-1,1]]))



