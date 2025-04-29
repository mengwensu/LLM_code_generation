import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from queue import PriorityQueue

def plan_shortest_path(grid_size, obstacles, start, end):
    def is_point_valid(point):
        x, y = point
        for obs in obstacles:
            (x1, y1), (x2, y2) = obs
            if x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5:
                return False
        return 0 <= x < grid_size and 0 <= y < grid_size

    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def a_star_path(start, end):
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        while not open_set.empty():
            _, current = open_set.get()

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in
                         [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]]

            for neighbor in neighbors:
                if not is_point_valid(neighbor):
                    continue

                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    open_set.put((f_score[neighbor], neighbor))

        return []  # No path found

    def draw_grid():
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(range(grid_size + 1))
        plt.yticks(range(grid_size + 1))
        return fig, ax

    def draw_obstacles(ax):
        for obs in obstacles:
            (x1, y1), (x2, y2) = obs
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='black', alpha=0.7))

    def draw_path(ax, path):
        if len(path) > 1:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='blue', linewidth=2, label='Path')

    def draw_points(ax):
        ax.scatter(*start, color='green', s=100, label='Start')
        ax.scatter(*end, color='red', s=100, label='End')

    path = a_star_path(start, end)

    fig, ax = draw_grid()
    draw_obstacles(ax)
    draw_path(ax, path)
    draw_points(ax)

    ax.legend()
    plt.show()

    if path and path[-1] == end:
        print("Path successfully found!")
    else:
        print("Path could not reach the goal. Incomplete path shown.")

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 1), (17, 3))]
start = (0, 0)
end = (19, 19)
plan_shortest_path(20, obstacles, start, end)
