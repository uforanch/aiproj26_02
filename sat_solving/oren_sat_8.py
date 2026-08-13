import sys, math, time
from itertools import product
sys.path.insert(0,'/mnt/data')
from oren_sat import (
    Hyperplane, cube_edges, _vertex_dot_products, _set_bit_indices,
    _remove_dominated_candidates, _hard_coverage_clauses, _solve_with_at_most_k
)

n=8
edges=cube_edges(n)
allmask=(1<<len(edges))-1
best={}

def add(normal, offset=0):
    normal=tuple(normal)
    dots=_vertex_dot_products(normal,(-1,1))
    cov=0
    for e in edges:
        v1=dots[e.lower_vertex]-offset
        v2=dots[e.upper_vertex]-offset
        if v1*v2<0:
            cov |= 1<<e.index
    if not cov: return
    h=Hyperplane(normal,offset,cov)
    old=best.get(cov)
    if old is None or (sum(map(abs,h.normal)),h.normal) < (sum(map(abs,old.normal)),old.normal):
        best[cov]=h

# Structured Q6 family on x1,...,x6: (p,p,p,q,q,r), bias 0.
for p,q,r in product(range(-8,9), repeat=3):
    if p==q==r==0: continue
    g=0
    for z in (p,q,r): g=math.gcd(g,abs(z))
    if g != 1: continue
    vec=(p,p,p,q,q,r,0,0)
    # identify +/- duplicates
    first=next(z for z in vec if z)
    if first<0: continue
    add(vec,0)

# Coordinate planes, especially x7=0 and x8=0.
for i in range(n):
    v=[0]*n; v[i]=1
    add(v,0)

cands=list(best.values())
cands=_remove_dominated_candidates(cands,len(edges))
cands.sort(key=lambda h:(-h.number_of_sliced_edges,h.normal,h.offset))
print('edges',len(edges))
print('candidates',len(cands))
clauses=_hard_coverage_clauses(cands,len(edges))
for k in (6,7):
    t=time.time()
    sol=_solve_with_at_most_k(clauses,len(cands),k,'cadical195')
    print('k',k,'SAT' if sol is not None else 'UNSAT','seconds',round(time.time()-t,3))
    if sol is not None:
        cov=0
        for idx in sol: cov |= cands[idx].coverage
        print('covered',cov.bit_count(),'of',len(edges))
        for j,idx in enumerate(sol,1):
            h=cands[idx]
            print(f'H{j}: {h.equation()} [{h.number_of_sliced_edges} edges]')