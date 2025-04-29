import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from heapq import heappush, heappop

# Helper function to check if a point is inside an obstacle
def is_point_in_obstacle(point, obstacles):
    x, y = point
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

# Helper function to check if a point is too close to an obstacle
def is_point_near_obstacle(point, obstacles, gap=0.5):
    x, y = point
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if (x1 - gap <= x <= x2 + gap) and (y1 - gap <= y <= y2 + gap):
            return True
    return False

# Helper function to get neighbors of a point (including diagonal movements)
def get_neighbors(point):
    x, y = point
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 20 and 0 <= new_y < 20:
                neighbors.append((new_x, new_y))
    return neighbors

# A* pathfinding algorithm
def a_star(start, goal, obstacles):
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current):
            if is_point_in_obstacle(neighbor, obstacles) or is_point_near_obstacle(neighbor, obstacles):
                continue

            tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Function to plan the path using Voronoi diagram
def plan_path(start, goal, obstacles):
    # Generate Voronoi diagram
    points = []
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        points.append((x1, y1))
        points.append((x1, y2))
        points.append((x2, y1))
        points.append((x2, y2))
    vor = Voronoi(points)

    # Find the safest path using A* on the Voronoi diagram
    path = a_star(start, goal, obstacles)

    return path, vor

# Function to plot the grid, obstacles, and path
def plot_grid(start, goal, obstacles, path, vor):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True)

    # Plot obstacles
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', alpha=0.5))

    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.5)

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color='orange', linewidth=2, marker='o', markersize=5)

    # Plot start and goal points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    ax.legend()
    plt.show()

# Main function to plan and plot the path
def plan_and_plot_path(start, goal, obstacles):
    path, vor = plan_path(start, goal, obstacles)
    plot_grid(start, goal, obstacles, path, vor)

# Example usage
start = (1, 1)
goal = (18, 18)
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 15)), ((7, 7), (9, 9))]
plan_and_plot_path(start, goal, obstacles)