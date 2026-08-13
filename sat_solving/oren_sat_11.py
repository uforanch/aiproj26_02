from __future__ import annotations

from itertools import permutations, product
from time import perf_counter

from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

N = 11
K = 9
BIAS2 = 1  # twice the bias 0.5; evaluate 2*a.x + BIAS2 exactly

# Last-four coefficient patterns from the known Q_11 construction.
TAILS = [
    (-3, 46, -13, 2),
    (2, -50, 11, -3),
    (-21, -3, -3, -41),
    (33, 2, 1, 28),
    (5, -14, -41, 4),
    (-30, -8, 4, 28),
    (36, 8, -4, -22),
    (-3, 12, 48, -4),
    (-20, 8, -8, 6),
]


def cube_edges(n: int):
    edges = []
    for coordinate in range(n):
        bit = 1 << coordinate
        for lower in range(1 << n):
            if lower & bit:
                continue
            edges.append((lower, lower | bit, coordinate))
    return edges


def vertex_values(normal: tuple[int, ...]):
    # Vertex bit 0 means -1 and bit 1 means +1.
    base = -sum(normal)
    vals = [0] * (1 << N)
    vals[0] = 2 * base + BIAS2
    for mask in range(1, 1 << N):
        lb = mask & -mask
        i = lb.bit_length() - 1
        vals[mask] = vals[mask ^ lb] + 4 * normal[i]
    return vals


def coverage_mask(normal: tuple[int, ...], edges):
    vals = vertex_values(normal)
    mask = 0
    for j, (u, v, _) in enumerate(edges):
        if vals[u] * vals[v] < 0:  # strict interior crossing only
            mask |= 1 << j
    return mask


def equation(normal: tuple[int, ...]) -> str:
    terms = []
    for i, a in enumerate(normal, 1):
        if not a:
            continue
        atom = f"x{i}" if abs(a) == 1 else f"{abs(a)}x{i}"
        if not terms:
            terms.append(atom if a > 0 else f"-{atom}")
        else:
            terms.append((" + " if a > 0 else " - ") + atom)
    # Stored form is a.x + 1/2 = 0, equivalently a.x = -1/2.
    return "".join(terms) + " = -1/2"


def main():
    start = perf_counter()
    edges = cube_edges(N)

    # Structured family: first seven coefficients are all -9 and the four
    # tail coordinates may be permuted. This includes the known Q_11 solution.
    normals = set()
    for tail in TAILS:
        for perm in set(permutations(tail)):
            normals.add((-9,) * 7 + tuple(perm))

    # Keep one representative per edge-coverage pattern.
    by_coverage = {}
    for normal in normals:
        cov = coverage_mask(normal, edges)
        if cov:
            by_coverage.setdefault(cov, normal)

    candidates = [(normal, cov) for cov, normal in by_coverage.items()]
    candidates.sort(key=lambda item: (-item[1].bit_count(), item[0]))

    clauses = [[] for _ in edges]
    for var, (_, cov) in enumerate(candidates, 1):
        bits = cov
        while bits:
            lb = bits & -bits
            clauses[lb.bit_length() - 1].append(var)
            bits ^= lb

    missing = [i for i, clause in enumerate(clauses) if not clause]
    if missing:
        raise RuntimeError(f"candidate family misses edges, first indices: {missing[:10]}")

    card = CardEnc.atmost(
        lits=list(range(1, len(candidates) + 1)),
        bound=K,
        top_id=len(candidates),
        encoding=EncType.seqcounter,
    )

    build_done = perf_counter()
    with Solver(name="cadical195", bootstrap_with=clauses + card.clauses) as solver:
        sat = solver.solve()
        solve_done = perf_counter()
        model = solver.get_model() if sat else None

    print(f"Dimension: {N}")
    print(f"Edges: {len(edges)}")
    print(f"Raw structured normals: {len(normals)}")
    print(f"Distinct coverage candidates: {len(candidates)}")
    print(f"k = {K}: {'SAT' if sat else 'UNSAT'}")
    print(f"Build time: {build_done - start:.3f} seconds")
    print(f"SAT time: {solve_done - build_done:.3f} seconds")

    if not sat:
        return

    positive = {lit for lit in model if lit > 0}
    selected = [candidates[i - 1] for i in range(1, len(candidates) + 1) if i in positive]
    union = 0
    for _, cov in selected:
        union |= cov

    print(f"Selected hyperplanes: {len(selected)}")
    print(f"Covered: {union.bit_count()} of {len(edges)} edges")
    for i, (normal, cov) in enumerate(selected, 1):
        print(f"H{i}: {equation(normal)}  [slices {cov.bit_count()} edges]")

    if union.bit_count() != len(edges):
        raise RuntimeError("SAT model failed exact strict-interior verification")


if __name__ == "__main__":
    main()