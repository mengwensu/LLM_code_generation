import numpy as np
import matplotlib.pyplot as plt
import random
import math

def is_collision_free(point, obstacles, gap=0.5):
    x, y = point
    for (x1, y1), (x2, y2) in obstacles:
        if x1 - gap <= x <= x2 + gap and y1 - gap <= y <= y2 + gap:
            return False
    return True

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def nearest_node(nodes, random_point):
    return min(nodes, key=lambda node: distance(node, random_point))

def steer(from_node, to_node, step_size=1.0):
    if distance(from_node, to_node) < step_size:
        return to_node
    else:
        theta = math.atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
        return from_node[0] + step_size * math.cos(theta), from_node[1] + step_size * math.sin(theta)

def rrt(start, goal, obstacles, max_iter=1000, step_size=1.0):
    nodes = [start]
    edges = []
    path = []

    for _ in range(max_iter):
        random_point = (random.uniform(0, 20), random.uniform(0, 20))
        nearest = nearest_node(nodes, random_point)
        new_node = steer(nearest, random_point, step_size)

        if is_collision_free(new_node, obstacles):
            nodes.append(new_node)
            edges.append((nearest, new_node))

            if distance(new_node, goal) < step_size:
                nodes.append(goal)
                edges.append((new_node, goal))
                path = [goal]
                current = new_node
                while current != start:
                    path.append(current)
                    for edge in edges:
                        if edge[1] == current:
                            current = edge[0]
                            break
                path.append(start)
                path.reverse()
                break

    return nodes, edges, path

def plot_grid(obstacles, start, goal, nodes, edges, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True)

    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray')
        ax.add_patch(rect)

    for edge in edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'bo-', markersize=2)

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'ro-', markersize=4)

    ax.plot(start[0], start[1], 'go', markersize=8)
    ax.plot(goal[0], goal[1], 'ro', markersize=8)

    plt.show()

def plan_path(obstacles, start, goal):
    nodes, edges, path = rrt(start, goal, obstacles)
    plot_grid(obstacles, start, goal, nodes, edges, path)

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12))]
start = (1, 1)
goal = (19, 19)

plan_path(obstacles, start, goal)
