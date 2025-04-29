import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def is_collision_free(node, obstacles):
    for (x1, y1), (x2, y2) in obstacles:
        if x1 <= node.x <= x2 and y1 <= node.y <= y2:
            return False
    return True

def nearest_node(nodes, random_node):
    return min(nodes, key=lambda node: distance(node, random_node))

def steer(from_node, to_node, step_size=1.0):
    if distance(from_node, to_node) < step_size:
        return to_node
    else:
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        return Node(from_node.x + step_size * math.cos(theta), from_node.y + step_size * math.sin(theta))

def rrt(start, goal, obstacles, max_iter=1000, step_size=1.0):
    nodes = [start]
    for _ in range(max_iter):
        random_node = Node(random.uniform(0, 20), random.uniform(0, 20))
        nearest = nearest_node(nodes, random_node)
        new_node = steer(nearest, random_node, step_size)

        if is_collision_free(new_node, obstacles):
            new_node.parent = nearest
            nodes.append(new_node)

            if distance(new_node, goal) <= step_size:
                goal.parent = new_node
                nodes.append(goal)
                return nodes
    return None

def backtrack(node):
    path = []
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def plot_grid_and_path(obstacles, path, start, goal):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True)

    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray')
        ax.add_patch(rect)

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'b-', linewidth=2)

    plt.plot(start.x, start.y, 'go', markersize=10)
    plt.plot(goal.x, goal.y, 'ro', markersize=10)
    plt.show()

def plan_path(obstacles, start, goal):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    nodes = rrt(start_node, goal_node, obstacles)
    if nodes:
        path = backtrack(goal_node)
        plot_grid_and_path(obstacles, path, start_node, goal_node)
    else:
        print("Path not found")

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)

plan_path(obstacles, start, end)
