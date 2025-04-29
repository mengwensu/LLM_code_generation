import numpy as np
import matplotlib.pyplot as plt
import heapq

def is_within_bounds(x, y, grid_size):
    return 0 <= x < grid_size and 0 <= y < grid_size

def is_obstacle(x, y, obstacles):
    for (x1, y1), (x2, y2) in obstacles:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def dijkstra(grid_size, start, end, obstacles):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    pq = [(0, start)]
    visited = set()
    distances = {start: 0}
    parent = {start: None}

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        if current_vertex == end:
            break

        for direction in directions:
            neighbor = (current_vertex[0] + direction[0], current_vertex[1] + direction[1])
            if is_within_bounds(neighbor[0], neighbor[1], grid_size) and not is_obstacle(neighbor[0], neighbor[1], obstacles):
                old_cost = distances.get(neighbor, float('inf'))
                new_cost = current_distance + heuristic(current_vertex, neighbor)
                if new_cost < old_cost:
                    distances[neighbor] = new_cost
                    parent[neighbor] = current_vertex
                    heapq.heappush(pq, (new_cost, neighbor))

    path = []
    step = end
    while step is not None:
        path.append(step)
        step = parent.get(step, None)
    path.reverse()
    return path

def plot_grid(grid_size, obstacles, path, start, end):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, grid_size+1, 1))
    ax.set_yticks(np.arange(0, grid_size+1, 1))
    ax.grid(which='both')

    for (x1, y1), (x2, y2) in obstacles:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='gray'))

    for (x, y) in path:
        ax.plot(x, y, 'bo')

    ax.plot(start[0], start[1], 'go')
    ax.plot(end[0], end[1], 'ro')

    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plan_shortest_path(grid_size, obstacles, start, end):
    path = dijkstra(grid_size, start, end, obstacles)
    plot_grid(grid_size, obstacles, path, start, end)
    return path

# Example usage
grid_size = 20
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)

path = plan_shortest_path(grid_size, obstacles, start, end)
print("Planned Path:", path)
