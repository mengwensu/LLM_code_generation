import heapq
import matplotlib.pyplot as plt
import numpy as np

def is_within_bounds(x, y, grid_size):
    return 0 <= x < grid_size and 0 <= y < grid_size

def is_obstacle(x, y, obstacles):
    for (x1, y1), (x2, y2) in obstacles:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(x, y, grid_size):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_within_bounds(nx, ny, grid_size):
            neighbors.append((nx, ny))
    return neighbors

def dijkstra(start, end, grid_size, obstacles):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current[0], current[1], grid_size):
            if is_obstacle(neighbor[0], neighbor[1], obstacles):
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def plot_grid(grid_size, obstacles, path, start, end):
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(which='both')

    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)

    ax.plot(start[0], start[1], 'go')
    ax.plot(end[0], end[1], 'ro')

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-')

    plt.show()

def plan_path(obstacles, start, end, grid_size=20):
    path = dijkstra(start, end, grid_size, obstacles)
    plot_grid(grid_size, obstacles, path, start, end)

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)

plan_path(obstacles, start, end)
