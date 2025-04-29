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

# A* pathfinding algorithm
def a_star(start, goal, obstacles, grid_size=20):
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                if is_point_in_obstacle(neighbor, obstacles) or is_point_near_obstacle(neighbor, obstacles):
                    continue
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

# Main function to plan the path
def plan_path(start, goal, obstacles):
    grid_size = 20
    path = a_star(start, goal, obstacles, grid_size)

    # Plotting
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True)

    # Plot obstacles
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', alpha=0.5))

    # Plot start and goal points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    else:
        print("No path found!")

    ax.legend()
    plt.show()

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 15), (17, 17))]
start = (1, 1)
goal = (18, 18)
plan_path(start, goal, obstacles)