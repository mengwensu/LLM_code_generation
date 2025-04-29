import numpy as np
import heapq
import matplotlib.pyplot as plt

def is_valid(x, y, grid, obstacles):
    """
    Check if the (x, y) position is valid, considering boundaries and obstacles.
    """
    if 0 <= x < 20 and 0 <= y < 20:
        for (x1, y1), (x2, y2) in obstacles:
            if x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5:
                return False
        return True
    return False

def dijkstra(grid, start, end, obstacles):
    """
    Implement Dijkstra's algorithm to find the shortest path.
    """
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
    ]
    
    # Priority queue: (cost, x, y, path)
    pq = [(0, start[0], start[1], [])]
    visited = set()

    while pq:
        cost, x, y, path = heapq.heappop(pq)
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Add the current position to the path
        path = path + [(x, y)]

        # Check if the end point is reached
        if (x, y) == end:
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, grid, obstacles) and (nx, ny) not in visited:
                heapq.heappush(pq, (cost + np.sqrt(dx**2 + dy**2), nx, ny, path))

    return []  # Return empty if no path found

def plot_grid(grid, obstacles, path, start, end):
    """
    Plot the grid, obstacles, start and end points, and the planned path.
    """
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xticks(range(21))
    plt.yticks(range(21))
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    
    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        plt.fill_betweenx([y1, y2], x1, x2, color='gray', alpha=0.7)

    # Plot the path
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color='blue', linewidth=2, label='Planned Path')

    # Plot start and end points
    plt.scatter(*start, color='green', s=100, label='Start')
    plt.scatter(*end, color='red', s=100, label='End')

    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Path Planning on a 20x20 Grid')
    plt.show()

def plan_path(obstacles, start, end):
    """
    Plan the path on a 20x20 grid using Dijkstra's algorithm.
    """
    grid = np.zeros((20, 20))
    
    # Generate the path using Dijkstra's algorithm
    path = dijkstra(grid, start, end, obstacles)
    
    # Print details and plot the results
    print("Grid: 20x20")
    print(f"Obstacles: {obstacles}")
    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Planned Path: {path}")
    
    plot_grid(grid, obstacles, path, start, end)

# Example usage
grid_size = (20, 20)
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 1), (9, 3))]
start = (0, 0)
goal = (19, 19)

plan_path(obstacles, start, goal)