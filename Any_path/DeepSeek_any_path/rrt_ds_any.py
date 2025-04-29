import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import KDTree

# Helper function to check if a point is in collision with any obstacle
def is_collision_free(point, obstacles, gap=0.5):
    x, y = point
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if (x1 - gap <= x <= x2 + gap) and (y1 - gap <= y <= y2 + gap):
            return False
    return True

# Helper function to generate a random point in the grid
def random_point():
    return (np.random.uniform(0, 20), np.random.uniform(0, 20))

# Helper function to find the nearest node in the tree to a given point
def nearest_node(tree, point):
    tree_array = np.array(list(tree.keys()))
    kdtree = KDTree(tree_array)
    dist, idx = kdtree.query(point)
    return tuple(tree_array[idx])

# Helper function to steer from one point towards another
def steer(from_point, to_point, step_size=1.0):
    direction = np.arctan2(to_point[1] - from_point[1], to_point[0] - from_point[0])
    new_x = from_point[0] + step_size * np.cos(direction)
    new_y = from_point[1] + step_size * np.sin(direction)
    return (new_x, new_y)

# RRT algorithm implementation
def rrt(start, goal, obstacles, max_iter=1000, step_size=1.0):
    tree = {start: None}  # Tree stored as a dictionary with child: parent pairs
    for _ in range(max_iter):
        rand_point = random_point()
        nearest = nearest_node(tree, rand_point)
        new_point = steer(nearest, rand_point, step_size)
        
        if is_collision_free(new_point, obstacles):
            tree[new_point] = nearest
            if np.linalg.norm(np.array(new_point) - np.array(goal)) < step_size:
                tree[goal] = new_point
                return tree
    return tree

# Function to extract the path from the tree
def extract_path(tree, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = tree[current]
    return path[::-1]

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
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', alpha=0.5))
    
    # Plot start and goal points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    ax.legend()
    plt.show()

# Main function to plan and visualize the path
def plan_path(start, goal, obstacles):
    tree = rrt(start, goal, obstacles)
    path = extract_path(tree, goal)
    plot_grid(start, goal, obstacles, path)

# Example usage
start = (1, 1)
goal = (18, 18)
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 15)), ((7, 7), (9, 9))]
plan_path(start, goal, obstacles)