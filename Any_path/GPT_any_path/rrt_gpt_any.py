import matplotlib.pyplot as plt
import numpy as np
import random

def is_in_obstacle(point, obstacles, clearance=0.5):
    """
    Check if a point is inside any obstacle with the given clearance.
    """
    x, y = point
    for (bottom_left, top_right) in obstacles:
        if (bottom_left[0] - clearance <= x <= top_right[0] + clearance) and (bottom_left[1] - clearance <= y <= top_right[1] + clearance):
            return True
    return False

def generate_random_point(grid_size, obstacles, clearance=0.5):
    """
    Generate a random point within the grid that is not in any obstacle.
    """
    while True:
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        if not is_in_obstacle((x, y), obstacles, clearance):
            return (x, y)

def find_nearest(tree, point):
    """
    Find the nearest point in the tree to the given point.
    """
    return min(tree, key=lambda node: np.linalg.norm(np.array(node) - np.array(point)))

def extend_tree(tree, nearest, random_point, step_size=0.5):
    """
    Extend the tree from the nearest point towards the random point.
    """
    direction = np.array(random_point) - np.array(nearest)
    distance = np.linalg.norm(direction)
    if distance < step_size:
        new_point = random_point
    else:
        new_point = tuple(np.array(nearest) + step_size * direction / distance)
    return new_point

def rrt(grid_size, start, end, obstacles, max_iterations=1000, step_size=0.5):
    """
    Perform the RRT algorithm to find a path from start to end.
    """
    tree = [start]
    parent = {start: None}
    
    for _ in range(max_iterations):
        random_point = generate_random_point(grid_size, obstacles)
        nearest = find_nearest(tree, random_point)
        new_point = extend_tree(tree, nearest, random_point, step_size)
        
        if not is_in_obstacle(new_point, obstacles):
            tree.append(new_point)
            parent[new_point] = nearest
            
            if np.linalg.norm(np.array(new_point) - np.array(end)) < step_size:
                parent[end] = new_point
                tree.append(end)
                break
                
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path[::-1]

def plot_grid(grid_size, obstacles, path, start, end):
    """
    Plot the grid, obstacles, start and end points, and the path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot grid
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    
    # Plot obstacles
    for (bottom_left, top_right) in obstacles:
        rect = plt.Rectangle(bottom_left, 
                             top_right[0] - bottom_left[0], 
                             top_right[1] - bottom_left[1], 
                             color='black')
        ax.add_patch(rect)
    
    # Plot start and end points
    ax.plot(start[0], start[1], 'go', label='Start')  # Start in green
    ax.plot(end[0], end[1], 'ro', label='End')       # End in red
    
    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    ax.legend()
    plt.show()

def plan_path(grid_size, obstacles, start, end):
    """
    Plan a path on a grid using RRT.
    """
    path = rrt(grid_size, start, end, obstacles)
    print(f"Planned path: {path}")
    plot_grid(grid_size, obstacles, path, start, end)

# Example usage
grid_size = 20
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)

plan_path(grid_size, obstacles, start, end)
