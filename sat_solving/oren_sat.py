from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

#pip install python-sat
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver


@dataclass(frozen=True)
class Edge:
    index: int
    lower_vertex: int
    upper_vertex: int
    coordinate: int


@dataclass(frozen=True)
class Hyperplane:
    normal: tuple[int, ...]
    offset: int
    coverage: int

    @property
    def number_of_sliced_edges(self) -> int:
        return self.coverage.bit_count()

    def equation(self, variable: str = "x") -> str:
        terms: list[str] = []
        for i, coefficient in enumerate(self.normal, start=1):
            if coefficient == 0:
                continue
            magnitude = abs(coefficient)
            atom = f"{variable}{i}" if magnitude == 1 else f"{magnitude}{variable}{i}"
            if not terms:
                terms.append(atom if coefficient > 0 else f"-{atom}")
            else:
                terms.append((" + " if coefficient > 0 else " - ") + atom)
        left = "".join(terms) if terms else "0"
        return f"{left} = {self.offset}"


@dataclass(frozen=True)
class SlicingSolution:
    dimension: int
    vertex_values: tuple[int, int]
    coefficient_bound: int
    offset_bound: int | None
    candidate_count: int
    hyperplanes: tuple[Hyperplane, ...]

    @property
    def size(self) -> int:
        return len(self.hyperplanes)


def cube_edges(dimension: int) -> list[Edge]:
    """Return all n*2^(n-1) edges, using bit 0/1 for low/high coordinates."""
    edges: list[Edge] = []
    index = 0
    for coordinate in range(dimension):
        bit = 1 << coordinate
        for vertex in range(1 << dimension):
            if vertex & bit:
                continue
            edges.append(
                Edge(
                    index=index,
                    lower_vertex=vertex,
                    upper_vertex=vertex | bit,
                    coordinate=coordinate,
                )
            )
            index += 1
    return edges


def _first_nonzero_is_positive(values: Sequence[int]) -> bool:
    for value in values:
        if value:
            return value > 0
    return False


def _primitive(normal: Sequence[int], offset: int) -> bool:
    divisor = abs(offset)
    for coefficient in normal:
        divisor = math.gcd(divisor, abs(coefficient))
    return divisor == 1


def _vertex_dot_products(
    normal: Sequence[int], vertex_values: tuple[int, int]
) -> list[int]:
    """Compute aÂ·v for every cube vertex, indexed by its high-coordinate bit mask."""
    low, high = vertex_values
    dimension = len(normal)
    delta = high - low
    dots = [0] * (1 << dimension)
    dots[0] = low * sum(normal)
    for mask in range(1, 1 << dimension):
        least_bit = mask & -mask
        coordinate = least_bit.bit_length() - 1
        dots[mask] = dots[mask ^ least_bit] + delta * normal[coordinate]
    return dots


def enumerate_candidate_hyperplanes(
    dimension: int,
    coefficient_bound: int,
    *,
    vertex_values: tuple[int, int] = (-1, 1),
    offset_bound: int | None = None,
    remove_dominated: bool = True,
) -> tuple[list[Hyperplane], list[Edge]]:
    """
    Enumerate primitive integer hyperplanes aÂ·x=b that slice at least one edge.

    The search is complete for primitive integer representations satisfying
    |a_i| <= coefficient_bound and, when supplied, |b| <= offset_bound.
    Hyperplanes differing only by multiplication by -1 are identified.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be at least 1")
    low, high = vertex_values
    if not isinstance(low, int) or not isinstance(high, int) or low >= high:
        raise ValueError("vertex_values must be two increasing integers")
    if offset_bound is not None and offset_bound < 0:
        raise ValueError("offset_bound must be nonnegative or None")

    edges = cube_edges(dimension)
    coefficient_values = range(-coefficient_bound, coefficient_bound + 1)

    # One representative per edge-coverage pattern is enough for minimum set cover.
    best_for_coverage: dict[int, Hyperplane] = {}

    for normal in product(coefficient_values, repeat=dimension):
        if not _first_nonzero_is_positive(normal):
            continue  # removes zero and the duplicate (-a,-b)

        dots = _vertex_dot_products(normal, vertex_values)
        coverage_by_offset: dict[int, int] = {}

        # An edge is sliced exactly when b lies strictly between its endpoint values.
        for edge in edges:
            endpoint_1 = dots[edge.lower_vertex]
            endpoint_2 = dots[edge.upper_vertex]
            lower_value = min(endpoint_1, endpoint_2) + 1
            upper_value = max(endpoint_1, endpoint_2) - 1
            if offset_bound is not None:
                lower_value = max(lower_value, -offset_bound)
                upper_value = min(upper_value, offset_bound)
            for offset in range(lower_value, upper_value + 1):
                coverage_by_offset[offset] = (
                    coverage_by_offset.get(offset, 0) | (1 << edge.index)
                )

        for offset, coverage in coverage_by_offset.items():
            if not _primitive(normal, offset):
                continue
            candidate = Hyperplane(tuple(normal), offset, coverage)
            old = best_for_coverage.get(coverage)
            if old is None:
                best_for_coverage[coverage] = candidate
            else:
                # Prefer a simpler displayed representative for the same coverage.
                old_score = (
                    max(max(map(abs, old.normal)), abs(old.offset)),
                    sum(map(abs, old.normal)) + abs(old.offset),
                    old.normal,
                    old.offset,
                )
                new_score = (
                    max(max(map(abs, candidate.normal)), abs(candidate.offset)),
                    sum(map(abs, candidate.normal)) + abs(candidate.offset),
                    candidate.normal,
                    candidate.offset,
                )
                if new_score < old_score:
                    best_for_coverage[coverage] = candidate

    candidates = list(best_for_coverage.values())
    if remove_dominated:
        candidates = _remove_dominated_candidates(candidates, len(edges))

    # A stable order makes runs reproducible.
    candidates.sort(
        key=lambda h: (
            -h.number_of_sliced_edges,
            max(max(map(abs, h.normal)), abs(h.offset)),
            h.normal,
            h.offset,
        )
    )
    return candidates, edges


def _remove_dominated_candidates(
    candidates: Sequence[Hyperplane], number_of_edges: int
) -> list[Hyperplane]:
    """Remove A when another candidate B covers every edge covered by A."""
    ordered = sorted(candidates, key=lambda h: h.coverage.bit_count(), reverse=True)
    kept: list[Hyperplane] = []
    kept_covering_edge: list[list[int]] = [[] for _ in range(number_of_edges)]

    for candidate in ordered:
        covered_edges = list(_set_bit_indices(candidate.coverage))
        if not covered_edges:
            continue

        # Every superset must contain every edge of candidate; inspect the rarest one.
        pivot = min(covered_edges, key=lambda edge: len(kept_covering_edge[edge]))
        dominated = False
        for kept_index in kept_covering_edge[pivot]:
            other = kept[kept_index]
            if candidate.coverage & other.coverage == candidate.coverage:
                dominated = True
                break
        if dominated:
            continue

        new_index = len(kept)
        kept.append(candidate)
        for edge in covered_edges:
            kept_covering_edge[edge].append(new_index)

    return kept


def _set_bit_indices(mask: int) -> Iterable[int]:
    while mask:
        least_bit = mask & -mask
        yield least_bit.bit_length() - 1
        mask ^= least_bit


def _greedy_cover(candidates: Sequence[Hyperplane], all_edges_mask: int) -> list[int]:
    uncovered = all_edges_mask
    selected: list[int] = []
    while uncovered:
        best_index = max(
            range(len(candidates)),
            key=lambda i: (candidates[i].coverage & uncovered).bit_count(),
        )
        gain = candidates[best_index].coverage & uncovered
        if not gain:
            missing = list(_set_bit_indices(uncovered))
            raise ValueError(f"candidate family cannot slice edges {missing[:10]}")
        selected.append(best_index)
        uncovered &= ~gain
    return selected


def _hard_coverage_clauses(
    candidates: Sequence[Hyperplane], number_of_edges: int
) -> list[list[int]]:
    clauses: list[list[int]] = [[] for _ in range(number_of_edges)]
    for candidate_index, candidate in enumerate(candidates, start=1):
        for edge_index in _set_bit_indices(candidate.coverage):
            clauses[edge_index].append(candidate_index)
    missing = [i for i, clause in enumerate(clauses) if not clause]
    if missing:
        raise ValueError(
            "No enumerated hyperplane slices some edges; increase the coefficient "
            f"bound. First missing edge indices: {missing[:10]}"
        )
    return clauses


def _solve_with_at_most_k(
    hard_clauses: Sequence[Sequence[int]],
    number_of_candidates: int,
    k: int,
    solver_name: str,
) -> list[int] | None:
    variables = list(range(1, number_of_candidates + 1))
    cardinality = CardEnc.atmost(
        lits=variables,
        bound=k,
        top_id=number_of_candidates,
        encoding=EncType.seqcounter,
    )
    clauses = [list(clause) for clause in hard_clauses]
    clauses.extend(cardinality.clauses)

    with Solver(name=solver_name, bootstrap_with=clauses) as solver:
        if not solver.solve():
            return None
        model = solver.get_model()
    positive = {literal for literal in model if literal > 0}
    return [i - 1 for i in variables if i in positive]


def _solve_minimum_maxsat(
    hard_clauses: Sequence[Sequence[int]],
    number_of_candidates: int,
    solver_name: str,
) -> list[int] | None:
    """Minimize the number of selected candidates using unit-weight partial MaxSAT."""
    formula = WCNF()
    for clause in hard_clauses:
        formula.append(list(clause))
    for variable in range(1, number_of_candidates + 1):
        formula.append([-variable], weight=1)

    with RC2(
        formula,
        solver=solver_name,
        adapt=True,
        exhaust=True,
        minz=True,
        trim=5,
    ) as optimizer:
        model = optimizer.compute()
        if model is None:
            return None

    positive = {literal for literal in model if literal > 0}
    return [i - 1 for i in range(1, number_of_candidates + 1) if i in positive]


def solve_hypercube_slicing(
    dimension: int,
    coefficient_bound: int,
    *,
    vertex_values: tuple[int, int] = (-1, 1),
    offset_bound: int | None = None,
    solver_name: str = "cadical195",
    remove_dominated: bool = True,
    target_k: int | None = None,
    optimization: str = "maxsat",
) -> SlicingSolution | None:
    """
    Find a minimum slicing family within the bounded candidate class.

    If target_k is supplied, use ordinary SAT to test whether a family of at most
    target_k exists. Otherwise, use partial MaxSAT by default to minimize the number
    selected; optimization="iterative-sat" instead uses repeated SAT calls.
    """
    candidates, edges = enumerate_candidate_hyperplanes(
        dimension,
        coefficient_bound,
        vertex_values=vertex_values,
        offset_bound=offset_bound,
        remove_dominated=remove_dominated,
    )
    if not candidates:
        return None

    hard_clauses = _hard_coverage_clauses(candidates, len(edges))
    all_edges_mask = (1 << len(edges)) - 1

    if target_k is not None:
        if target_k < 0:
            raise ValueError("target_k must be nonnegative")
        selected = _solve_with_at_most_k(
            hard_clauses, len(candidates), target_k, solver_name
        )
        if selected is None:
            return None
    elif optimization == "maxsat":
        selected = _solve_minimum_maxsat(
            hard_clauses, len(candidates), solver_name
        )
        if selected is None:
            return None
    elif optimization == "iterative-sat":
        max_single_cover = max(h.coverage.bit_count() for h in candidates)
        lower_bound = math.ceil(len(edges) / max_single_cover)
        upper_bound = len(_greedy_cover(candidates, all_edges_mask))

        low, high = lower_bound, upper_bound
        while low < high:
            middle = (low + high) // 2
            selected = _solve_with_at_most_k(
                hard_clauses, len(candidates), middle, solver_name
            )
            if selected is None:
                low = middle + 1
            else:
                high = middle

        selected = _solve_with_at_most_k(
            hard_clauses, len(candidates), low, solver_name
        )
        if selected is None:
            raise RuntimeError("unexpected SAT inconsistency")
    else:
        raise ValueError("optimization must be 'maxsat' or 'iterative-sat'")

    chosen = tuple(candidates[i] for i in selected)
    covered = 0
    for hyperplane in chosen:
        covered |= hyperplane.coverage
    if covered != all_edges_mask:
        raise RuntimeError("solver model failed exact coverage verification")

    return SlicingSolution(
        dimension=dimension,
        vertex_values=vertex_values,
        coefficient_bound=coefficient_bound,
        offset_bound=offset_bound,
        candidate_count=len(candidates),
        hyperplanes=chosen,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find bounded-integer hypercube slicing families with SAT."
    )
    parser.add_argument("-n", "--dimension", type=int, required=True)
    parser.add_argument(
        "-A",
        "--coefficient-bound",
        type=int,
        default=2,
        help="bound A in |a_i| <= A (default: 2)",
    )
    parser.add_argument(
        "-B",
        "--offset-bound",
        type=int,
        default=None,
        help="optional bound B in |b| <= B",
    )
    parser.add_argument(
        "--cube",
        choices=("pm1", "01"),
        default="pm1",
        help="use vertices {-1,+1}^n or {0,1}^n (default: pm1)",
    )
    parser.add_argument("--solver", default="cadical195")
    parser.add_argument(
        "--optimization",
        choices=("maxsat", "iterative-sat"),
        default="maxsat",
        help="exact minimization method when -k is omitted (default: maxsat)",
    )
    parser.add_argument(
        "-k",
        "--target-k",
        type=int,
        default=None,
        help="test existence with at most k hyperplanes instead of minimizing",
    )
    parser.add_argument(
        "--keep-dominated",
        action="store_true",
        help="disable safe set-cover dominance pruning",
    )
    args = parser.parse_args()

    vertex_values = (-1, 1) if args.cube == "pm1" else (0, 1)
    solution = solve_hypercube_slicing(
        dimension=args.dimension,
        coefficient_bound=args.coefficient_bound,
        vertex_values=vertex_values,
        offset_bound=args.offset_bound,
        solver_name=args.solver,
        remove_dominated=not args.keep_dominated,
        target_k=args.target_k,
        optimization=args.optimization,
    )

    if solution is None:
        qualifier = (
            f" with at most {args.target_k} hyperplanes" if args.target_k is not None else ""
        )
        print(f"No solution found{qualifier} within the coefficient bounds.")
        raise SystemExit(1)

    number_of_edges = args.dimension * (1 << (args.dimension - 1))
    print(f"Cube vertices: {solution.vertex_values}^n")
    print(f"Edges: {number_of_edges}")
    print(f"Candidates after deduplication/pruning: {solution.candidate_count}")
    print(f"Selected hyperplanes: {solution.size}")
    for index, hyperplane in enumerate(solution.hyperplanes, start=1):
        print(
            f"  H{index}: {hyperplane.equation()}  "
            f"[slices {hyperplane.number_of_sliced_edges} edges]"
        )


if __name__ == "__main__":
    main()