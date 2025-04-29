import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from heapq import heappop, heappush


def plan_shortest_path(grid_size, obstacles, start, end):
    """
    Plan the shortest path on a grid using Voronoi algorithm.

    :param grid_size: Tuple (width, height) of the grid (e.g., (20, 20)).
    :param obstacles: List of obstacles defined by bottom-left and top-right coordinates (e.g., [((3, 3), (5, 5))]).
    :param start: Tuple (x, y) for the starting point.
    :param end: Tuple (x, y) for the ending point.
    """
    def create_obstacle_mask(grid_size, obstacles):
        mask = np.zeros(grid_size, dtype=bool)
        for (x1, y1), (x2, y2) in obstacles:
            mask[x1:x2 + 1, y1:y2 + 1] = True
        return mask

    def voronoi_nodes_with_clearance(mask, clearance):
        points = []
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if not mask[x, y]:
                    points.append((x, y))
        return points

    def dijkstra_path(vor_nodes, start, end):
        open_set = []
        heappush(open_set, (0, start))
        visited = {}
        visited[start] = None
        cost_so_far = {start: 0}

        while open_set:
            current_cost, current = heappop(open_set)

            if current == end:
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in vor_nodes:
                    new_cost = current_cost + np.hypot(dx, dy)
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost
                        heappush(open_set, (priority, neighbor))
                        visited[neighbor] = current

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = visited[current]
        path.reverse()

        return path

    # Create obstacle mask
    mask = create_obstacle_mask(grid_size, obstacles)

    # Generate Voronoi nodes with clearance
    clearance = 1  # Minimum clearance distance
    vor_points = voronoi_nodes_with_clearance(mask, clearance)

    # Convert points to a Voronoi diagram
    vor = Voronoi(vor_points)

    # Ensure start and end are valid
    if start not in vor_points or end not in vor_points:
        raise ValueError("Start or end point is not valid (may overlap with obstacles).")

    # Find shortest path using Dijkstra's algorithm
    path = dijkstra_path(set(vor_points), start, end)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(np.arange(0, grid_size[0] + 1, 1))
    ax.set_yticks(np.arange(0, grid_size[1] + 1, 1))
    ax.grid(which="both")

    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, color="gray", alpha=0.7)
        ax.add_patch(rect)

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color="blue", linewidth=2, label="Path")
    else:
        print("No valid path found.")

    # Highlight start and end
    ax.plot(start[0], start[1], "go", label="Start")
    ax.plot(end[0], end[1], "ro", label="End")

    plt.legend()
    plt.show()

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 15), (18, 18))]
start = (0, 0)
end = (19, 19)
grid_size = (20, 20)

plan_shortest_path(grid_size, obstacles, start, end)
