import matplotlib.pyplot as plt
import random
import math

def is_collision_free(node, obstacles):
    """Check if a node is in a collision-free space."""
    x, y = node
    for (x1, y1), (x2, y2) in obstacles:
        if x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5:
            return False
    return True

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_nearest_node(tree, point):
    """Find the nearest node in the tree to the given point."""
    return min(tree, key=lambda n: distance(n, point))

def steer(from_node, to_node, step_size):
    """Move from from_node towards to_node by step_size."""
    x1, y1 = from_node
    x2, y2 = to_node
    theta = math.atan2(y2 - y1, x2 - x1)
    x_new = x1 + step_size * math.cos(theta)
    y_new = y1 + step_size * math.sin(theta)
    return round(x_new, 1), round(y_new, 1)

def is_path_collision_free(from_node, to_node, obstacles):
    """Check if the straight path between two nodes is collision-free."""
    steps = int(distance(from_node, to_node) / 0.1)
    for i in range(steps + 1):
        intermediate_point = steer(from_node, to_node, i * 0.1)
        if not is_collision_free(intermediate_point, obstacles):
            return False
    return True

def rrt(start, goal, obstacles, max_iterations=1000, step_size=1.0):
    """Plan a path using the RRT algorithm."""
    tree = [start]
    parents = {start: None}
    for _ in range(max_iterations):
        random_point = (random.uniform(0, 20), random.uniform(0, 20))
        nearest_node = get_nearest_node(tree, random_point)
        new_node = steer(nearest_node, random_point, step_size)
        if is_collision_free(new_node, obstacles) and is_path_collision_free(nearest_node, new_node, obstacles):
            tree.append(new_node)
            parents[new_node] = nearest_node
            if distance(new_node, goal) <= step_size and is_path_collision_free(new_node, goal, obstacles):
                parents[goal] = new_node
                tree.append(goal)
                break
    else:
        goal = None  # Goal not reached

    # Extract path
    path = []
    current = goal if goal else get_nearest_node(tree, goal)
    while current:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path

def plot_grid(start, goal, obstacles, path):
    """Plot the grid, obstacles, start, goal, and path."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    ax.grid(True)

    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color="gray"))

    # Plot start and goal
    ax.plot(*start, "go", label="Start")
    ax.plot(*goal, "ro", label="Goal")

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, "-b", label="Path")

    ax.legend()
    plt.show()

def plan_path(start, goal, obstacles):
    """Main function to plan the shortest path on a 20x20 grid using RRT."""
    path = rrt(start, goal, obstacles)
    plot_grid(start, goal, obstacles, path)
    if path and path[-1] == goal:
        print("Path found:", path)
    else:
        print("Goal not reached. Path up to the furthest point:", path)

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (13, 13)), ((15, 2), (18, 4))]
start = (1, 1)
goal = (19, 19)
plan_path(start, goal, obstacles)
