import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import heapq

def is_valid_point(x, y, obstacles, grid_size=20):
    """Check if a point is valid (within grid and not in obstacles with buffer)"""
    if not (0 <= x <= grid_size and 0 <= y <= grid_size):
        return False
    
    # Check buffer zone of 0.5 around obstacles
    for (x1, y1), (x2, y2) in obstacles:
        if (x1 - 0.5 <= x <= x2 + 0.5 and 
            y1 - 0.5 <= y <= y2 + 0.5):
            return False
    return True

def get_neighbors(x, y, obstacles):
    """Get all possible neighbors including diagonal moves"""
    moves = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # cardinal directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal directions
    ]
    neighbors = []
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if is_valid_point(new_x, new_y, obstacles):
            # Diagonal moves have cost sqrt(2), cardinal moves have cost 1
            cost = 1.414 if dx != 0 and dy != 0 else 1
            neighbors.append((new_x, new_y, cost))
    return neighbors

def heuristic(x1, y1, x2, y2):
    """Calculate Euclidean distance as heuristic"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def plan_path(start, end, obstacles):
    """Plan shortest path using A* with Voronoi considerations"""
    start_x, start_y = start
    end_x, end_y = end
    
    # Generate Voronoi points from obstacle corners
    voronoi_points = []
    for (x1, y1), (x2, y2) in obstacles:
        voronoi_points.extend([(x1, y1), (x2, y1), (x1, y2), (x2, y2)])
    voronoi_points = list(set(voronoi_points))  # Remove duplicates
    
    if not voronoi_points:
        voronoi_points.append(start)  # Add start point if no obstacles
    
    vor = Voronoi(np.array(voronoi_points))
    
    # A* algorithm
    open_set = [(0, start_x, start_y)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start_x, start_y, end_x, end_y)}
    
    furthest_point = start
    min_dist_to_goal = heuristic(start_x, start_y, end_x, end_y)
    
    while open_set:
        current_f, current_x, current_y = heapq.heappop(open_set)
        current = (current_x, current_y)
        
        # Update furthest point reached
        dist_to_goal = heuristic(current_x, current_y, end_x, end_y)
        if dist_to_goal < min_dist_to_goal:
            min_dist_to_goal = dist_to_goal
            furthest_point = current
        
        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], True
        
        for next_x, next_y, cost in get_neighbors(current_x, current_y, obstacles):
            next_point = (next_x, next_y)
            tentative_g = g_score[current] + cost
            
            if next_point not in g_score or tentative_g < g_score[next_point]:
                came_from[next_point] = current
                g_score[next_point] = tentative_g
                f_score[next_point] = tentative_g + heuristic(next_x, next_y, end_x, end_y)
                heapq.heappush(open_set, (f_score[next_point], next_x, next_y))
    
    # If goal not reached, return path to furthest point
    path = []
    current = furthest_point
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1], False

def visualize_path(start, end, obstacles, path, success):
    """Visualize the grid, obstacles, and path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up grid
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    
    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1-0.5, y1-0.5), x2-x1+1, y2-y1+1, 
                           facecolor='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # Plot start and end points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(end[0], end[1], 'ro', markersize=10, label='End')
    
    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    ax.legend()
    plt.title("Path Planning" + (" (Incomplete)" if not success else ""))
    plt.show()
    
    # Print details
    print("Grid: 20x20")
    print("Obstacles:", obstacles)
    print("Start:", start)
    print("End:", end)
    print("Path:", path)
    print("Goal Reached:", success)

def main():
    # Example usage
    start = (2, 2)
    end = (18, 18)
    obstacles = [((3, 3), (5, 5)), ((8, 8), (10, 10)), ((12, 5), (14, 7))]
    
    path, success = plan_path(start, end, obstacles)
    visualize_path(start, end, obstacles, path, success)

if __name__ == "__main__":
    main()



    