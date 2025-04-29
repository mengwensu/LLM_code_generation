import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import heapq

def plot_grid_and_obstacles(obstacles):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(which='both')

    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    return fig, ax

def compute_voronoi(obstacles):
    points = []
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        points.extend([(x1, y1), (x2, y1), (x1, y2), (x2, y2)])

    vor = Voronoi(points)
    return vor

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(start, goal, vor):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, vor):
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def get_neighbors(node, vor):
    neighbors = []
    for region in vor.regions:
        if node in region:
            neighbors.extend(region)
    return neighbors

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def plot_path(ax, path):
    path_line = LineCollection([(path[i], path[i+1]) for i in range(len(path) - 1)], colors='blue', linewidths=2)
    ax.add_collection(path_line)

def plan_path(obstacles, start, end):
    fig, ax = plot_grid_and_obstacles(obstacles)
    vor = compute_voronoi(obstacles)
    path = astar(start, end, vor)

    if path:
        plot_path(ax, path)
        ax.plot(start[0], start[1], 'go')  # Start point in green
        ax.plot(end[0], end[1], 'ro')  # End point in red

    plt.show()

obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)

plan_path(obstacles, start, end)
