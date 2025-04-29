import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import KDTree

# Helper function to check if a point is within an obstacle
def is_point_in_obstacle(point, obstacles):
    x, y = point
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

# Helper function to check if a line segment intersects with any obstacle
def is_line_in_obstacle(start, end, obstacles):
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if (start[0] >= x1 - 0.5 and start[0] <= x2 + 0.5 and
            start[1] >= y1 - 0.5 and start[1] <= y2 + 0.5) or \
           (end[0] >= x1 - 0.5 and end[0] <= x2 + 0.5 and
            end[1] >= y1 - 0.5 and end[1] <= y2 + 0.5):
            return True
    return False

# RRT algorithm implementation
def rrt(start, goal, obstacles, max_iter=1000, step_size=1.0):
    tree = KDTree([start])
    nodes = [start]
    parents = {0: -1}

    for i in range(max_iter):
        random_point = (np.random.uniform(0, 20), np.random.uniform(0, 20))
        nearest_idx = tree.query(random_point)[1]
        nearest_node = nodes[nearest_idx]

        direction = np.arctan2(random_point[1] - nearest_node[1], random_point[0] - nearest_node[0])
        new_point = (nearest_node[0] + step_size * np.cos(direction),
                     nearest_node[1] + step_size * np.sin(direction))

        if not is_line_in_obstacle(nearest_node, new_point, obstacles):
            nodes.append(new_point)
            tree = KDTree(nodes)
            parents[len(nodes) - 1] = nearest_idx

            if np.linalg.norm(np.array(new_point) - np.array(goal)) < step_size:
                nodes.append(goal)
                tree = KDTree(nodes)
                parents[len(nodes) - 1] = len(nodes) - 2
                break

    path = []
    current_idx = len(nodes) - 1
    while current_idx != -1:
        path.append(nodes[current_idx])
        current_idx = parents[current_idx]
    path.reverse()

    return path

# Function to plot the grid, obstacles, and path
def plot_grid(start, goal, obstacles, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True)

    # Plot obstacles
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray'))

    # Plot start and goal points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')

    ax.legend()
    plt.show()

# Main function to plan the path and plot the grid
def plan_path(start, goal, obstacles):
    path = rrt(start, goal, obstacles)
    plot_grid(start, goal, obstacles, path)

# Example usage
start = (1, 1)
goal = (18, 18)
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 15)), ((7, 7), (9, 9))]
plan_path(start, goal, obstacles)