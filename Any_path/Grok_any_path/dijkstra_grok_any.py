import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def is_valid_point(x, y, obstacles, grid_size=20):
    """Check if a point is valid (within grid and not in obstacle)"""
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        return False
    
    # Check if point is too close to obstacles (maintain 0.5 buffer)
    for (x1, y1), (x2, y2) in obstacles:
        if (x1 - 0.5 <= x <= x2 + 0.5 and 
            y1 - 0.5 <= y <= y2 + 0.5):
            return False
    return True

def get_neighbors(x, y, obstacles, grid_size=20):
    """Get valid neighboring points including diagonals"""
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # horizontal/vertical
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
    ]
    neighbors = []
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if is_valid_point(new_x, new_y, obstacles, grid_size):
            # Distance is 1 for cardinal directions, √2 for diagonals
            dist = 1.414 if dx != 0 and dy != 0 else 1
            neighbors.append((new_x, new_y, dist))
    return neighbors

def dijkstra_path(start, end, obstacles, grid_size=20):
    """Find shortest path using Dijkstra's algorithm"""
    start_x, start_y = start
    end_x, end_y = end
    
    if not (is_valid_point(start_x, start_y, obstacles) and 
            is_valid_point(end_x, end_y, obstacles)):
        return None
    
    distances = {(start_x, start_y): 0}
    pq = [(0, start_x, start_y)]
    previous = {}
    visited = set()
    
    while pq:
        dist, x, y = heapq.heappop(pq)
        
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        if (x, y) == (end_x, end_y):
            break
            
        for next_x, next_y, step_dist in get_neighbors(x, y, obstacles, grid_size):
            if (next_x, next_y) in visited:
                continue
                
            new_dist = dist + step_dist
            
            if (next_x, next_y) not in distances or new_dist < distances[(next_x, next_y)]:
                distances[(next_x, next_y)] = new_dist
                previous[(next_x, next_y)] = (x, y)
                heapq.heappush(pq, (new_dist, next_x, next_y))
    
    # Reconstruct path
    if (end_x, end_y) not in previous and (end_x, end_y) != (start_x, start_y):
        # Find furthest reachable point if goal not reached
        furthest = start
        max_dist = 0
        for point, dist in distances.items():
            if dist > max_dist:
                max_dist = dist
                furthest = point
    
    path = []
    current = (end_x, end_y) if (end_x, end_y) in previous or (end_x, end_y) == (start_x, start_y) else furthest
    
    while current in previous or current == (start_x, start_y):
        path.append(current)
        if current == (start_x, start_y):
            break
        current = previous.get(current)
    
    return list(reversed(path))

def plot_path(start, end, obstacles, path):
    """Create interactive plot of the grid, obstacles, and path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set grid
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    
    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        rect = patches.Rectangle(
            (x1-0.5, y1-0.5), (x2-x1+1), (y2-y1+1),
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5
        )
        ax.add_patch(rect)
    
    # Plot start and end points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(end[0], end[1], 'ro', markersize=10, label='End')
    
    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    ax.legend()
    plt.title("Path Planning with Dijkstra's Algorithm")
    plt.show()

def plan_path(start, end, obstacles):
    """Main function to plan and visualize path"""
    print("Grid: 20x20")
    print("Start point:", start)
    print("End point:", end)
    print("Obstacles:", obstacles)
    
    path = dijkstra_path(start, end, obstacles)
    
    if path:
        print("Planned path:", path)
        if path[-1] != end:
            print("Warning: Goal not reached. Showing path to furthest reachable point.")
    else:
        print("No valid path found!")
    
    plot_path(start, end, obstacles, path)
    return path

# Example usage
if __name__ == "__main__":
    start_point = (2, 2)
    end_point = (18, 18)
    obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((8, 15), (15, 17))]
    
    plan_path(start_point, end_point, obstacles)



