import heapq
import matplotlib.pyplot as plt
import numpy as np

# Helper function to check if a point is within an obstacle
def is_within_obstacle(point, obstacles, gap=0.5):
    x, y = point
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        if (x1 - gap <= x <= x2 + gap) and (y1 - gap <= y <= y2 + gap):
            return True
    return False

# Helper function to get neighbors, including diagonal movements
def get_neighbors(point, obstacles, gap=0.5):
    x, y = point
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 20 and 0 <= new_y < 20:
                if not is_within_obstacle((new_x, new_y), obstacles, gap):
                    neighbors.append((new_x, new_y))
    return neighbors

# Dijkstra's algorithm implementation
def dijkstra(start, end, obstacles, gap=0.5):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        current_cost, current = heapq.heappop(open_set)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(current, obstacles, gap):
            tentative_g_score = g_score[current] + 1  # Assuming each step has a cost of 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))
    
    # If the goal is not reached, return the path up to the furthest point
    furthest_point = max(g_score.keys(), key=lambda x: g_score[x])
    path = []
    current = furthest_point
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]

# Function to plot the grid, obstacles, and path
def plot_grid(start, end, obstacles, path):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_yticks(np.arange(0, 21, 1))
    ax.grid(True)
    
    # Plot obstacles
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, marker='o', color='blue', label='Path')
    
    # Plot start and end points
    ax.scatter(*start, color='green', s=100, label='Start')
    ax.scatter(*end, color='red', s=100, label='End')
    
    ax.legend()
    plt.show()

# Main function to plan the path and plot the grid
def plan_path(start, end, obstacles):
    path = dijkstra(start, end, obstacles)
    plot_grid(start, end, obstacles, path)

# Example usage
start = (1, 1)
end = (18, 18)
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((15, 15), (17, 17))]
plan_path(start, end, obstacles)