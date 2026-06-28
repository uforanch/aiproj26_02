import torch
from src import *
import pytest

def test_count_p1():
    t1_matrix = torch.tensor([[1,1,1,3,3,-4,0],
                           [-2,-2,-2,3,3,-1,0],
                           [3,3,3,1,1,-4,0],
                           [-1,-1,-1,3,3,6,0],
                           [3,3,3,1,1,8,0]]).reshape(35).double()
    hpoints1, hpoints2 = gen_hypercubes(6)
    edge_set = torch.tensor(np.concatenate((hpoints1, hpoints2)).T).double()


    assert count(6,t1_matrix, edge_set)==192.0

    t1_matrix = torch.tensor([[-9,-9,-9,-9,-9,-9,7,-16,5,35,.5],
                           [-9,-9,-9,-9,-9,-9,-32,-4,-17,8,.5],
                           [-9,-9,-9,-9,-9,-9,32,5,19,-4,.5],
                           [-9,-9,-9,-9,-9,-9,-3,15,-3,-38,.5],
                           [-9,-9,-9,-9,-9,-9,15,3,-36,4,.5],
                           [-9,-9,-9,-9,-9,-9,8,-35,-2,-12,.5],
                           [-9,-9,-9,-9,-9,-9,-4,33,7,16,.5],
                           [-9,-9,-9,-9,-9,-9,-18,-4,34,-5,.5]]).reshape(88).double()

    hpoints1, hpoints2 = gen_hypercubes(10)
    edge_set = torch.tensor(np.concatenate((hpoints1, hpoints2)).T).double()

    assert count(10,t1_matrix,edge_set)==5120.0

def test_count_p2():
    t1_matrix = torch.tensor([[1,3,-4,0],
                           [-2,3,-1,0],
                           [3,1,-4,0],
                           [-1,3,6,0],
                           [3,1,8,0]]).double()
    t1_matrix = torch.flatten(t1_matrix)

    hpoints1, hpoints2,p_count=gen_reducehypercubes(6,(3,2,1))
    p_count = torch.tensor(p_count).double()
    edge_set = torch.tensor(np.concatenate((hpoints1, hpoints2)).T).double()

    assert count_reduced((3,2,1),t1_matrix, edge_set,p_count)==192.0
    t1_matrix = torch.tensor([[-9,7,-16,5,35,.5],
                           [-9,-32,-4,-17,8,.5],
                           [-9,32,5,19,-4,.5],
                           [-9,-3,15,-3,-38,.5],
                           [-9,15,3,-36,4,.5],
                           [-9,8,-35,-2,-12,.5],
                           [-9,-4,33,7,16,.5],
                           [-9,-18,-4,34,-5,.5]]).double()
    t1_matrix = torch.flatten(t1_matrix)

    hpoints1, hpoints2, p_count = gen_reducehypercubes(10, (6,1,1,1,1))
    p_count = torch.tensor(p_count).double()
    edge_set = torch.tensor(np.concatenate((hpoints1, hpoints2)).T).double()

    assert count_reduced((6,1,1,1,1), t1_matrix, edge_set, p_count)  == 5120.0

def test_count_basic():
    e_list = torch.tensor([2,0,-1,0]).double()
    h = torch.tensor([1,-1,0]).double()

    assert count(2,h,e_list) == 1.0
    h = torch.tensor([1,-1,3]).double()
    assert count(2,h,e_list) == 0.0
    h = torch.tensor([[1,-1,0],[1,-1,0]]).double()
    h = torch.flatten(h)
    assert count(2,h,e_list) == 1.0

    h = torch.tensor([[1,-1,3],[1,-1,3]]).double()
    h = torch.flatten(h)
    assert count(2,h,e_list) == 0.0
    hp1, hp2 = gen_hypercubes(2)
    e_list = torch.tensor(np.concatenate((hp1, hp2)).T).double()
    h = torch.tensor([1,0,.5]).double()
    assert count(2,h,e_list) == 2.0
    h = torch.tensor([[1, 0, .5], [1,-1,.5]]).double()
    h = torch.flatten(h)
    assert count(2,h,e_list)  == 3.0

@pytest.mark.parametrize("d", [3,4,5])
def test_non_reduced_hyperplanes(d):
    hpoints1, hpoints2 = gen_hypercubes(d, no_loading=True)
    s=set()
    h_len = hpoints1.shape[1]
    for i1 in range(h_len):
        t1=tuple(hpoints1[:,i1])
        t2=tuple(hpoints2[:,i1])
        s.add((t1,t2))
        s.add((t2, t1))
    assert 2**d*d == len(s)

def test_reduced_hyperplanes():
    hpoints1, hpoints2, _  = gen_reducehypercubes(6,(3,2,1), no_loading=True)
    correct_dict = {(-3,-2,-1):3,
                    (-3,-2,1):2,
                    (-3,0,-1):3,
                    (-1,-2,-1):3,
                    (-3,0,1):2,
                    (-1,-2,1):2,
                    (1,2,1):1,
                    (3,0,1):1,
                    (3,2,-1):1,
                    (3,2,1):0}
    points = set()
    edges = Counter()
    l = hpoints1.shape[1]
    for i1 in range(l):
        p1 = tuple(hpoints1[:,i1])
        p2 = tuple(hpoints2[:,i1])
        points.add(p1)
        points.add(p2)
        edges[(p1,p2)]+=1
    for k,v in edges.items():
        if k in correct_dict:
            assert correct_dict[k] == v