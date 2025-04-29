import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.patches import Rectangle


def plan_rrt(grid_size, start, end, obstacles):
    """
    Plans the shortest path using the RRT algorithm on a 20x20 grid.
    Args:
        grid_size (int): Size of the grid (e.g., 20 for a 20x20 grid).
        start (tuple): Starting point (x, y).
        end (tuple): Ending point (x, y).
        obstacles (list): List of rectangular obstacles as ((x1, y1), (x2, y2)).
    """
    def is_collision_free(point, obstacles):
        """Check if a point is free of collision with any obstacle."""
        for obs in obstacles:
            x1, y1 = obs[0]
            x2, y2 = obs[1]
            if x1 - 0.5 <= point[0] <= x2 + 0.5 and y1 - 0.5 <= point[1] <= y2 + 0.5:
                return False
        return True

    def find_nearest(tree, random_point):
        """Find the nearest point in the tree to a given random point."""
        return min(tree, key=lambda node: np.linalg.norm(np.array(node) - np.array(random_point)))

    def steer(from_node, to_node, step_size=1.0):
        """Move from one node towards another by step size."""
        direction = np.array(to_node) - np.array(from_node)
        length = np.linalg.norm(direction)
        if length == 0:
            return from_node
        step = step_size * direction / length
        new_node = tuple(np.array(from_node) + step)
        return (round(new_node[0], 2), round(new_node[1], 2))

    # Initialize the tree with the starting point
    tree = [start]
    parent_map = {start: None}
    goal_reached = False
    path = []

    while not goal_reached:
        # Generate a random point within the grid
        random_point = (random.uniform(0, grid_size), random.uniform(0, grid_size))

        # Find the nearest node in the tree
        nearest_node = find_nearest(tree, random_point)

        # Move towards the random point
        new_node = steer(nearest_node, random_point)

        # Check if the new node is collision-free
        if is_collision_free(new_node, obstacles):
            tree.append(new_node)
            parent_map[new_node] = nearest_node

            # Check if we are within a step of the goal
            if np.linalg.norm(np.array(new_node) - np.array(end)) <= 1.0:
                goal_reached = True
                parent_map[end] = new_node
                path.append(end)

    # Backtrack to find the path
    current = end
    while current:
        path.append(current)
        current = parent_map[current]

    path.reverse()

    # Visualize the grid, obstacles, and path
    visualize_grid(grid_size, start, end, obstacles, path)


def visualize_grid(grid_size, start, end, obstacles, path):
    """Visualize the grid, obstacles, and path."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the grid
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)

    # Plot the start and end points
    ax.plot(*start, 'go', label='Start', markersize=10)
    ax.plot(*end, 'ro', label='End', markersize=10)

    # Plot the obstacles
    for obs in obstacles:
        x1, y1 = obs[0]
        x2, y2 = obs[1]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', alpha=0.7)
        ax.add_patch(rect)

    # Plot the path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', label='Path', linewidth=2)
        ax.plot(path_x, path_y, 'bo', markersize=5)

    # Add legend
    ax.legend()
    plt.show()

# Example usage
grid_size = 20
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 0), (18, 3))]
start = (0, 0)
goal = (19, 19)

plan_rrt(grid_size, start, goal, obstacles)
