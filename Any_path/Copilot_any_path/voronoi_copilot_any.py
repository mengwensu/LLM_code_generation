import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Voronoi

def generate_voronoi_grid(obstacles, start, end):
    points = [(x, y) for x in range(1, 20) for y in range(1, 20)]
    for (x1, y1), (x2, y2) in obstacles:
        points = [p for p in points if not (x1-0.5 <= p[0] <= x2+0.5 and y1-0.5 <= p[1] <= y2+0.5)]
    
    vor = Voronoi(points)
    return vor

def create_graph_from_voronoi(vor, obstacles):
    G = nx.Graph()
    
    for v_idx, vertex in enumerate(vor.vertices):
        x, y = vertex
        if 0 <= x <= 20 and 0 <= y <= 20:
            if not any((x1-0.5 <= x <= x2+0.5 and y1-0.5 <= y <= y2+0.5) for (x1, y1), (x2, y2) in obstacles):
                G.add_node(v_idx, pos=(x, y))
    
    for (v1, v2) in vor.ridge_vertices:
        if v1 >= 0 and v2 >= 0 and v1 in G.nodes and v2 in G.nodes:
            G.add_edge(v1, v2, weight=np.linalg.norm(np.array(vor.vertices[v1]) - np.array(vor.vertices[v2])))
    
    return G

def find_nearest_node(graph, pos):
    return min(graph.nodes, key=lambda n: np.linalg.norm(np.array(graph.nodes[n]['pos']) - np.array(pos)))

def find_path(graph, start, end):
    start_node = find_nearest_node(graph, start)
    end_node = find_nearest_node(graph, end)
    try:
        path_nodes = nx.shortest_path(graph, start_node, end_node, weight='weight')
        return [graph.nodes[n]['pos'] for n in path_nodes]
    except nx.NetworkXNoPath:
        return []

def plot_grid(vor, obstacles, path, start, end):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    for simplex in vor.ridge_vertices:
        simplex = [v for v in simplex if v >= 0]
        if len(simplex) == 2:
            p1, p2 = vor.vertices[simplex]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5)
    
    for (x1, y1), (x2, y2) in obstacles:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='black', alpha=0.7))
    
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Planned Path')
    
    plt.scatter(*start, color='green', s=100, label='Start')
    plt.scatter(*end, color='red', s=100, label='End')
    
    plt.legend()
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.show()

def plan_path(obstacles, start, end):
    vor = generate_voronoi_grid(obstacles, start, end)
    graph = create_graph_from_voronoi(vor, obstacles)
    path = find_path(graph, start, end)
    plot_grid(vor, obstacles, path, start, end)
    return path

# Example Usage:
obstacles = [((3, 3), (5, 5)), ((8, 8), (10, 10)), ((13, 2), (15, 6))]
start = (1, 1)
end = (18, 18)
plan_path(obstacles, start, end)
