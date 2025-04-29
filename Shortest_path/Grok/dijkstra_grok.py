import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def is_valid_point(x, y, obstacles):
    """Check if a point is valid (within grid and not in obstacles)"""
    if not (0 <= x <= 20 and 0 <= y <= 20):
        return False
    
    for (bottom_left, top_right) in obstacles:
        bl_x, bl_y = bottom_left
        tr_x, tr_y = top_right
        if bl_x <= x <= tr_x and bl_y <= y <= tr_y:
            return False
    return True

def get_neighbors(x, y, obstacles):
    """Get valid neighboring points including diagonals"""
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # cardinal directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal directions
    ]
    neighbors = []
    
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if is_valid_point(new_x, new_y, obstacles):
            # Calculate distance (1 for cardinal, sqrt(2) for diagonal)
            dist = 1.414 if dx != 0 and dy != 0 else 1
            neighbors.append((new_x, new_y, dist))
    return neighbors

def dijkstra_path(start, end, obstacles):
    """Find shortest path using Dijkstra's algorithm"""
    start_x, start_y = start
    end_x, end_y = end
    
    # Check if start and end points are valid
    if not (is_valid_point(start_x, start_y, obstacles) and 
            is_valid_point(end_x, end_y, obstacles)):
        return None
    
    # Priority queue for Dijkstra's
    queue = [(0, start_x, start_y, [start])]
    visited = set()
    
    while queue:
        dist, x, y, path = heapq.heappop(queue)
        
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        if (x, y) == (end_x, end_y):
            return path
            
        for next_x, next_y, move_dist in get_neighbors(x, y, obstacles):
            if (next_x, next_y) not in visited:
                new_dist = dist + move_dist
                new_path = path + [(next_x, next_y)]
                heapq.heappush(queue, (new_dist, next_x, next_y, new_path))
    
    # If no path found, return the path to the furthest point reached
    if path:
        return path
    return None

def visualize_grid(start, end, obstacles, path=None):
    """Visualize the grid with obstacles and path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set grid
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    
    # Plot obstacles
    for (bl, tr) in obstacles:
        bl_x, bl_y = bl
        tr_x, tr_y = tr
        width = tr_x - bl_x
        height = tr_y - bl_y
        ax.add_patch(Rectangle((bl_x, bl_y), width, height, 
                             facecolor='gray', alpha=0.5))
    
    # Plot start and end points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(end[0], end[1], 'ro', markersize=10, label='End')
    
    # Plot path if exists
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    ax.legend()
    plt.title("Shortest Path Planning")
    plt.show()

def plan_shortest_path(start, end, obstacles):
    """Main function to plan and visualize shortest path"""
    # Print grid information
    print("Grid: 20x20")
    print(f"Start point: {start}")
    print(f"End point: {end}")
    print("Obstacles (bottom-left, top-right):")
    for obs in obstacles:
        print(f"  {obs}")
    
    # Find path
    path = dijkstra_path(start, end, obstacles)
    
    # Print results
    if path:
        print("\nPath found:")
        print(" -> ".join([str(p) for p in path]))
    else:
        print("\nNo complete path found")
        if path:
            print("Furthest path reached:", " -> ".join([str(p) for p in path]))
    
    # Visualize
    visualize_grid(start, end, obstacles, path)

# Example usage
if __name__ == "__main__":
    start_point = (2, 2)
    end_point = (18, 18)
    obstacles = [((3, 3), (5, 5)), ((8, 8), (10, 12)), ((15, 5), (17, 7))]
    
    plan_shortest_path(start_point, end_point, obstacles)