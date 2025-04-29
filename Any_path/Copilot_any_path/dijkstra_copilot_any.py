import heapq
import numpy as np
import matplotlib.pyplot as plt

def is_valid(x, y, obstacles, buffer=0.5):
    """Check if a coordinate is valid (not inside or too close to an obstacle)."""
    for (x1, y1), (x2, y2) in obstacles:
        if x1 - buffer <= x <= x2 + buffer and y1 - buffer <= y <= y2 + buffer:
            return False
    return 0 <= x < 20 and 0 <= y < 20

def dijkstra(start, end, obstacles):
    """Finds the shortest path using Dijkstra's algorithm."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Horizontal & Vertical
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal
    
    pq = [(0, start)]  # Priority queue (distance, node)
    distances = {start: 0}
    previous = {}
    visited = set()
    
    while pq:
        dist, (x, y) = heapq.heappop(pq)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        if (x, y) == end:
            break  # Goal reached
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, obstacles) and (nx, ny) not in visited:
                new_dist = dist + (1.4 if dx != 0 and dy != 0 else 1)  # Diagonal cost = 1.4
                if new_dist < distances.get((nx, ny), float('inf')):
                    distances[(nx, ny)] = new_dist
                    previous[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_dist, (nx, ny)))
    
    # Reconstruct the path
    path, node = [], end
    while node in previous:
        path.append(node)
        node = previous[node]
    path.append(start)
    return path[::-1] if end in visited else path[::-1]  # Return full or partial path

def plot_grid(start, end, obstacles, path):
    """Plot the 20x20 grid with obstacles and the planned path."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, color='gray', alpha=0.6)
        ax.add_patch(rect)
    
    # Plot path
    if path:
        px, py = zip(*path)
        ax.plot(px, py, marker='o', markersize=3, color='blue', linestyle='-')
    
    # Plot start and end points
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*end, color='red', s=100, label='End')
    ax.legend()
    plt.show()

# Example Usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 15), (9, 18))]
start = (0, 0)
end = (19, 19)
path = dijkstra(start, end, obstacles)
plot_grid(start, end, obstacles, path)
