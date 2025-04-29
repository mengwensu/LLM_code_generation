import heapq
import matplotlib.pyplot as plt
import numpy as np

# Define the grid size
GRID_SIZE = 20

# Define movement directions (8 directions: up, down, left, right, and 4 diagonals)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

def is_within_grid(x, y):
    """Check if a point is within the grid boundaries."""
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def is_obstacle_free(x, y, obstacles):
    """Check if a point is not inside any obstacle."""
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if x1 <= x <= x2 and y1 <= y <= y2:
            return False
    return True

def is_path_clear(x1, y1, x2, y2, obstacles):
    """Check if the path between two points is clear of obstacles."""
    # Check if the midpoint is clear (simple check for diagonal movement)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    return is_obstacle_free(mid_x, mid_y, obstacles)

def dijkstra(start, end, obstacles):
    """Dijkstra's algorithm to find the shortest path on a grid."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, (x, y) = heapq.heappop(open_set)

        if (x, y) == end:
            break

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not is_within_grid(nx, ny):
                continue
            if not is_obstacle_free(nx, ny, obstacles):
                continue
            if not is_path_clear(x, y, nx, ny, obstacles):
                continue

            new_cost = cost_so_far[(x, y)] + np.sqrt(dx**2 + dy**2)
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                priority = new_cost
                heapq.heappush(open_set, (priority, (nx, ny)))
                came_from[(nx, ny)] = (x, y)

    # Reconstruct path
    path = []
    current = end
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    return path

def plot_grid(start, end, obstacles, path):
    """Plot the grid, obstacles, and path."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.grid(True)

    # Plot obstacles
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', alpha=0.5)
        ax.add_patch(rect)

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, marker='o', color='blue', linestyle='-', linewidth=2, markersize=5)

    # Plot start and end points
    ax.plot(start[0], start[1], marker='o', color='green', markersize=10, label='Start')
    ax.plot(end[0], end[1], marker='o', color='red', markersize=10, label='End')

    ax.legend()
    plt.show()

def plan_path(start, end, obstacles):
    """Plan the shortest path on a 20x20 grid using Dijkstra's algorithm."""
    path = dijkstra(start, end, obstacles)
    plot_grid(start, end, obstacles, path)

# Example usage
start = (1, 1)
end = (18, 18)
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 15), (17, 17))]

plan_path(start, end, obstacles)