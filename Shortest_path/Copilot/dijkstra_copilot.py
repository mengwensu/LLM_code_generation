import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush

def dijkstra_shortest_path(grid_size, obstacles, start, end):
    """
    Plans the shortest path using Dijkstra's algorithm on a 20x20 grid.
    Visualizes the grid, obstacles, and planned path.
    """
    def is_valid(x, y):
        """Checks if a point is within bounds and not inside an obstacle."""
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            return False
        for (bl, tr) in obstacles:
            if bl[0] - 0.5 <= x <= tr[0] + 0.5 and bl[1] - 0.5 <= y <= tr[1] + 0.5:
                return False
        return True

    def get_neighbors(x, y):
        """Gets all valid neighboring points including diagonal movements."""
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(came_from, current):
        """Reconstructs the path from the end point to the start point."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    # Dijkstra's algorithm
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current_cost, current = heappop(open_set)
        if current == end:
            break

        for neighbor in get_neighbors(*current):
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor) - np.array(current))
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                priority = tentative_g_score
                heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    path = reconstruct_path(came_from, end) if end in g_score else reconstruct_path(came_from, max(g_score, key=g_score.get))
    return path, path[-1] == end

def visualize_grid(grid_size, obstacles, path, start, end, success):
    """Visualizes the grid, obstacles, and planned path."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.grid(True)

    # Plot obstacles
    for (bl, tr) in obstacles:
        ax.add_patch(plt.Rectangle((bl[0] - 0.5, bl[1] - 0.5),
                                   tr[0] - bl[0] + 1, tr[1] - bl[1] + 1,
                                   color='black', alpha=0.7))

    # Plot start and end points
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*end, color='red', s=100, label='End')

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color='blue', linewidth=2, label='Path')
    if not success:
        ax.scatter(*path[-1], color='orange', s=100, label='Furthest Point')

    ax.legend()
    plt.show()

def plan_shortest_path(grid_size, obstacles, start, end):
    """Main function to plan the shortest path and visualize the grid."""
    path, success = dijkstra_shortest_path(grid_size, obstacles, start, end)
    visualize_grid(grid_size, obstacles, path, start, end, success)

# Example usage
grid_size = 20
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 0), (18, 3))]
start = (0, 0)
end = (19, 19)

plan_shortest_path(grid_size, obstacles, start, end)
