import matplotlib.pyplot as plt
import numpy as np
import heapq

def dijkstra_with_plot(obstacles, start, end):
    def is_within_grid(x, y):
        return 0 <= x < 20 and 0 <= y < 20

    def is_valid_move(x, y):
        for obs in obstacles:
            (x1, y1), (x2, y2) = obs
            if x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5:
                return False
        return is_within_grid(x, y)

    def get_neighbors(x, y):
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]
        return [(x + dx, y + dy) for dx, dy in directions if is_valid_move(x + dx, y + dy)]

    grid = np.zeros((20, 20))

    for obs in obstacles:
        (x1, y1), (x2, y2) = obs
        grid[y1:y2 + 1, x1:x2 + 1] = 1

    visited = set()
    pq = [(0, start)]  # Priority queue: (cost, (x, y))
    costs = {start: 0}
    parents = {start: None}

    while pq:
        current_cost, current = heapq.heappop(pq)
        
        if current in visited:
            continue

        visited.add(current)

        if current == end:
            break

        for neighbor in get_neighbors(*current):
            new_cost = current_cost + np.linalg.norm(np.array(neighbor) - np.array(current))

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                parents[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    path = []
    node = end

    while node is not None and node in parents:
        path.append(node)
        node = parents[node]

    path = path[::-1]

    if path[-1] != end:
        print("Goal not reached. Showing incomplete path.")

    # Plotting the grid
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    ax.grid(which="both", color="gray", linestyle="--", linewidth=0.5)

    # Plot obstacles
    for obs in obstacles:
        (x1, y1), (x2, y2) = obs
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color="black", alpha=0.7)
        ax.add_patch(rect)

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color="blue", linewidth=2, label="Path")

    # Mark start and end points
    ax.scatter(*start, color="green", s=100, label="Start")
    ax.scatter(*end, color="red", s=100, label="End")

    ax.legend()
    plt.show()

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
end = (19, 19)
dijkstra_with_plot(obstacles, start, end)
