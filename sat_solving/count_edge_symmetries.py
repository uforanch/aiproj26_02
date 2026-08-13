import networkx as nx
from itertools import permutations, product, combinations


def get_hypercube_edges(d):
    """Generates the full edge set of a d-dimensional {-1, 1}^d hypercube."""
    # Vertices are tuples of length d with values -1 and 1
    vertices = list(product([-1, 1], repeat=d))
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            # Edges exist between vertices differing by exactly one coordinate
            diff = sum(1 for x, y in zip(vertices[i], vertices[j]) if x != y)
            if diff == 1:
                edges.append((vertices[i], vertices[j]))
    return edges


def count_edge_subset_symmetries_vs_full(d, edge_subset):
    """
    Measures the number of hypercube symmetries that preserve the edge_subset.
    """
    # 1. Create the full hypercube graph
    Q_d = nx.Graph()
    Q_d.add_edges_from(get_hypercube_edges(d))

    # We represent the edge subset by coloring the edges:
    # 1 for edges in the subset, 0 for all other hypercube edges.
    edge_subset_set = {frozenset(edge) for edge in edge_subset}

    for u, v in Q_d.edges():
        if frozenset((u, v)) in edge_subset_set:
            Q_d[u][v]['color'] = 1
        else:
            Q_d[u][v]['color'] = 0

    # 2. To find automorphisms of Q_d that preserve edge colors,
    # we use GraphMatcher with an edge_match condition.
    def edge_match(e1, e2):
        return e1['color'] == e2['color']

    # Match Q_d to itself under the edge-color restriction
    matcher = nx.algorithms.isomorphism.GraphMatcher(Q_d, Q_d, edge_match=edge_match)

    # Count the valid isomorphisms
    symmetry_count = 0
    for _ in matcher.isomorphisms_iter():
        symmetry_count += 1

    return symmetry_count



def count_edge_subset_symmetries(edge_subset):
    """
    Measures the number of hypercube symmetries that preserve the edge_subset.
    """
    # 1. Create the full hypercube graph
    Q_d = nx.Graph()
    Q_d.add_edges_from(edge_subset)


    # Match Q_d to itself under the edge-color restriction
    matcher = nx.algorithms.isomorphism.GraphMatcher(Q_d, Q_d)

    # Count the valid isomorphisms
    symmetry_count = 0
    for _ in matcher.isomorphisms_iter():
        symmetry_count += 1

    return symmetry_count
# --- Example Usage ---
if __name__ == "__main__":
    d = 3  # 3D Hypercube (Cube)

    # Let's define a subset of edges.
    # For example, let's select a single face (a square) on the top hyperplane (z = 1).
    # Vertices on this face: (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)
    my_subset = [
        ((1, 1, 1), (-1, 1, 1)),
        ((-1, 1, 1), (-1, -1, 1)),
        ((-1, -1, 1), (1, -1, 1)),
        ((1, -1, 1), (1, 1, 1))
    ]

    total_cube_symmetries = 2 ** d * 6  # 2^3 * 3! = 48
    subset_symmetries = count_edge_subset_symmetries(my_subset)

    print(f"Dimension d = {d}")
    print(f"Total symmetries of the hypercube: {total_cube_symmetries}")
    print(f"Symmetries preserving the selected edge subset: {subset_symmetries}")
    print("\n(Note: A single face on a 3D cube has 8 symmetries: 4 rotations & 4 reflections.)")

