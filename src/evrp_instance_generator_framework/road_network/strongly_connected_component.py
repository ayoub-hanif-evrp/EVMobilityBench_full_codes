import networkx as nx


def largest_strongly_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """
    Extract the largest strongly connected component (SCC).

    This mirrors your benchmark goal: avoid dead-end directed structure that
    makes service reachability ambiguous.

    The raw OSM graph is downloaded with ``retain_all=True`` so small weak
    fragments inside the city boundary are not dropped before this step; SCC
    then picks one coherent drivable core.
    """

    components = list(nx.strongly_connected_components(G))
    if not components:
        raise ValueError("Graph has no strongly connected components.")

    largest = max(components, key=len)
    return G.subgraph(largest).copy()

