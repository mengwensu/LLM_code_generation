import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from queue import PriorityQueue

def create_grid(size):
    return np.zeros((size, size))

def add_obstacles(grid, obstacles):
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        grid[y1:y2+1, x1:x2+1] = 1
    return grid

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                length = np.linalg.norm(np.array(new_vertices[v2]) - np.array(new_vertices[v1]))
                p = np.array(new_vertices[v1]) + np.array(new_vertices[v2]) - np.array(new_vertices[v1])
                p /= length
                p = list(p * radius + center)
                new_region.append(len(new_vertices))
                new_vertices.append(p)
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

def plot_voronoi(vor, regions, vertices):
    fig = voronoi_plot_2d(vor)
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)
    plt.plot(vor.points[:, 0], vor.points[:, 1], 'o')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.gca().invert_yaxis()
    plt.show()

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def a_star_search(grid, start, goal, obstacles):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            next_cell = (current[0] + dx, current[1] + dy)
            if 0 <= next_cell[0] < grid.shape[0] and 0 <= next_cell[1] < grid.shape[1]:
                if grid[next_cell[1], next_cell[0]] == 0:
                    new_cost = cost_so_far[current] + heuristic(current, next_cell)
                    if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                        cost_so_far[next_cell] = new_cost
                        priority = new_cost + heuristic(goal, next_cell)
                        frontier.put(next_cell, priority)
                        came_from[next_cell] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def plot_path(grid, path, start, goal, obstacles):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='binary', origin='upper')
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(which='both')

    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    ax.plot(start[0], start[1], 'go')
    ax.plot(goal[0], goal[1], 'ro')

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-')

    plt.show()

def plan_path(obstacles, start, goal):
    grid = create_grid(20)
    grid = add_obstacles(grid, obstacles)

    points = [(x, y) for x in range(20) for y in range(20) if grid[y, x] == 0]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    plot_voronoi(vor, regions, vertices)

    came_from, cost_so_far = a_star_search(grid, start, goal, obstacles)
    path = reconstruct_path(came_from, start, goal)

    plot_path(grid, path, start, goal, obstacles)

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12))]
start = (1, 1)
goal = (18, 18)
plan_path(obstacles, start, goal)
