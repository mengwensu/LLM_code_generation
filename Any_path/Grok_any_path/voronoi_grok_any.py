import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from queue import PriorityQueue

def create_grid_with_obstacles(obstacles):
    """Create a 20x20 grid and mark obstacles"""
    grid = np.zeros((20, 20))
    for (x1, y1), (x2, y2) in obstacles:
        grid[x1:x2+1, y1:y2+1] = 1
    return grid

def is_valid_point(x, y, grid, obstacles):
    """Check if a point is valid (within bounds and not in obstacle with 0.5 buffer)"""
    if not (0 <= x <= 20 and 0 <= y <= 20):
        return False
    
    # Check buffer zone around obstacles
    for (x1, y1), (x2, y2) in obstacles:
        if (x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5):
            return False
    return True

def get_voronoi_points(obstacles, start, end):
    """Generate points for Voronoi diagram including obstacle corners and boundaries"""
    points = [start, end]
    # Add obstacle corners with 0.5 buffer
    for (x1, y1), (x2, y2) in obstacles:
        points.extend([
            (x1 - 0.5, y1 - 0.5), (x2 + 0.5, y1 - 0.5),
            (x1 - 0.5, y2 + 0.5), (x2 + 0.5, y2 + 0.5)
        ])
    # Add grid boundary points
    for i in range(0, 21, 5):
        points.extend([(i, 0), (i, 20), (0, i), (20, i)])
    return np.array(points)

def heuristic(a, b):
    """Calculate diagonal distance heuristic"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def plan_path(obstacles, start, end):
    """Plan path using Voronoi diagram and A* algorithm"""
    # Create grid and validate input
    grid = create_grid_with_obstacles(obstacles)
    if not (is_valid_point(start[0], start[1], grid, obstacles) and 
            is_valid_point(end[0], end[1], grid, obstacles)):
        print("Start or end point is invalid!")
        return None

    # Generate Voronoi diagram
    points = get_voronoi_points(obstacles, start, end)
    vor = Voronoi(points)

    # Possible movements (8 directions)
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]

    # A* algorithm
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}
    visited = set()

    while not frontier.empty():
        current = frontier.get()[1]
        
        if current == end:
            break
            
        if current in visited:
            continue
            
        visited.add(current)
        
        # Check neighboring points
        for dx, dy in directions:
            next_x = current[0] + dx
            next_y = current[1] + dy
            next_point = (next_x, next_y)
            
            if not is_valid_point(next_x, next_y, grid, obstacles):
                continue
                
            new_cost = cost_so_far[current] + heuristic(current, next_point)
            
            if next_point not in cost_so_far or new_cost < cost_so_far[next_point]:
                cost_so_far[next_point] = new_cost
                priority = new_cost + heuristic(next_point, end)
                frontier.put((priority, next_point))
                came_from[next_point] = current

    # Reconstruct path
    path = []
    current = end
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    # Print information
    print("Grid size: 20x20")
    print("Obstacles:", obstacles)
    print("Planned path:", path if end in came_from else f"Incomplete path to furthest point: {path}")

    # Visualize
    visualize_grid(obstacles, start, end, path, vor)
    
    return path if end in came_from else path

def visualize_grid(obstacles, start, end, path, vor):
    """Visualize the grid, obstacles, and path"""
    plt.figure(figsize=(10, 10))
    
    # Plot grid
    plt.grid(True)
    plt.xticks(range(21))
    plt.yticks(range(21))
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    
    # Plot obstacles
    for (x1, y1), (x2, y2) in obstacles:
        rect = plt.Rectangle((x1-0.5, y1-0.5), x2-x1+1, y2-y1+1, 
                           color='gray', alpha=0.5)
        plt.gca().add_patch(rect)
    
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, show_vertices=False, line_colors='gray', 
                    line_alpha=0.3)
    
    # Plot start and end points
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(end[0], end[1], 'ro', markersize=10, label='End')
    
    # Plot path
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
    
    plt.legend()
    plt.title("Voronoi-based Path Planning")
    plt.show()

# Example usage
if __name__ == "__main__":
    obstacles = [((3, 3), (5, 5)), ((10, 12), (12, 14))]
    start = (1, 1)
    end = (18, 18)
    path = plan_path(obstacles, start, end)