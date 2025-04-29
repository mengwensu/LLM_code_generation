import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_voronoi_points(grid_size, obstacles):
    """
    Generate points for the Voronoi diagram, avoiding obstacles.
    """
    points = []
    for x in range(grid_size):
        for y in range(grid_size):
            if is_valid_point(x, y, obstacles):
                points.append((x, y))
    return np.array(points)

def is_valid_point(x, y, obstacles):
    """
    Check if a point is valid by ensuring it's not inside or near an obstacle.
    """
    for (x1, y1), (x2, y2) in obstacles:
        if x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5:
            return False
    return True

def find_closest_voronoi_vertices(vor, point):
    """
    Find the closest Voronoi vertex to a given point.
    """
    distances = np.linalg.norm(vor.vertices - point, axis=1)
    return np.argmin(distances)

def find_voronoi_path(vor, start, end):
    """
    Find a path between the start and end points using Voronoi vertices and edges.
    """
    start_vertex = find_closest_voronoi_vertices(vor, np.array(start))
    end_vertex = find_closest_voronoi_vertices(vor, np.array(end))

    graph = build_voronoi_graph(vor)
    path = shortest_path(graph, start_vertex, end_vertex)
    return [tuple(vor.vertices[v]) for v in path]

def build_voronoi_graph(vor):
    """
    Build a graph from Voronoi ridge vertices.
    """
    graph = {}
    for ridge in vor.ridge_vertices:
        if all(v >= 0 for v in ridge):  # Skip invalid vertices
            v1, v2 = ridge
            graph.setdefault(v1, []).append(v2)
            graph.setdefault(v2, []).append(v1)
    return graph

def shortest_path(graph, start, end):
    """
    Use a simple BFS to find the shortest path in the graph.
    """
    from collections import deque
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current == end:
            return path
        for neighbor in graph.get(current, []):
            queue.append((neighbor, path + [neighbor]))
    return []

def plot_voronoi(grid_size, obstacles, path, vor, start, end):
    """
    Plot the Voronoi diagram, obstacles, start/end points, and the planned path.
    """
    plt.figure(figsize=(10, 10))
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)

    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        plt.fill_betweenx([y1, y2], x1, x2, color='gray', alpha=0.7)

    # Plot Voronoi diagram
    voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors='blue', ax=plt.gca())

    # Plot the path
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color='orange', linewidth=2, label='Planned Path')

    # Plot start and end points
    plt.scatter(*start, color='green', s=100, label='Start')
    plt.scatter(*end, color='red', s=100, label='End')

    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Voronoi Path Planning on a 20x20 Grid')
    plt.show()

def plan_path_with_voronoi(obstacles, start, end):
    """
    Plan a path using Voronoi's algorithm on a 20x20 grid.
    """
    grid_size = 20
    points = generate_voronoi_points(grid_size, obstacles)
    vor = Voronoi(points)

    # Find path using Voronoi vertices
    path = find_voronoi_path(vor, start, end)

    # Print and plot the results
    print("Grid: 20x20")
    print(f"Obstacles: {obstacles}")
    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Planned Path: {path}")

    plot_voronoi(grid_size, obstacles, path, vor, start, end)

obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)

plan_path_with_voronoi(obstacles, start, end)
