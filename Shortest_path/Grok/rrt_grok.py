import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import math

class RRTPlanner:
    def __init__(self, grid_size=20, step_size=1.0, max_iterations=5000):
        self.grid_size = grid_size
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.nodes = []
        self.edges = []
        self.safety_distance = 0.5

    def is_collision_free(self, point, obstacles):
        """Check if a point is collision-free with obstacles"""
        x, y = point
        # Check grid boundaries
        if x < 0 or x > self.grid_size or y < 0 or y > self.grid_size:
            return False
        
        # Check collision with obstacles
        for (bottom_left, top_right) in obstacles:
            bl_x, bl_y = bottom_left
            tr_x, tr_y = top_right
            if (x >= bl_x - self.safety_distance and x <= tr_x + self.safety_distance and
                y >= bl_y - self.safety_distance and y <= tr_y + self.safety_distance):
                return False
        return True

    def distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def nearest_node(self, point):
        """Find the nearest node in the tree"""
        return min(self.nodes, key=lambda node: self.distance(node, point))

    def steer(self, from_node, to_point):
        """Steer from a node towards a point with step size"""
        dist = self.distance(from_node, to_point)
        if dist <= self.step_size:
            return to_point
        theta = math.atan2(to_point[1] - from_node[1], to_point[0] - from_node[0])
        return (from_node[0] + self.step_size * math.cos(theta),
                from_node[1] + self.step_size * math.sin(theta))

    def plan_path(self, start, goal, obstacles):
        """Main RRT planning function"""
        self.nodes = [start]
        self.edges = []

        for _ in range(self.max_iterations):
            # Random point generation
            if random.random() < 0.1:  # Bias towards goal
                rand_point = goal
            else:
                rand_point = (random.uniform(0, self.grid_size),
                            random.uniform(0, self.grid_size))

            # Find nearest node and create new node
            nearest = self.nearest_node(rand_point)
            new_node = self.steer(nearest, rand_point)

            # Check collision
            if self.is_collision_free(new_node, obstacles):
                self.nodes.append(new_node)
                self.edges.append((nearest, new_node))

                # Check if goal is reached
                if self.distance(new_node, goal) <= self.step_size:
                    if self.is_collision_free(goal, obstacles):
                        self.nodes.append(goal)
                        self.edges.append((new_node, goal))
                        return True
        
        return False

    def reconstruct_path(self, start, goal):
        """Reconstruct the path from start to goal"""
        if goal not in self.nodes:
            # Find closest point to goal if goal wasn't reached
            goal = min(self.nodes, key=lambda node: self.distance(node, goal))
        
        path = [goal]
        current = goal
        
        while current != start:
            for edge in self.edges:
                if edge[1] == current:
                    current = edge[0]
                    path.append(current)
                    break
        
        return list(reversed(path))

def plot_path(start, goal, obstacles, path, planner):
    """Plot the grid, obstacles, and path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set grid and axes
    ax.set_xlim(0, planner.grid_size)
    ax.set_ylim(0, planner.grid_size)
    ax.grid(True)
    ax.set_xticks(range(planner.grid_size + 1))
    ax.set_yticks(range(planner.grid_size + 1))

    # Plot obstacles
    for (bottom_left, top_right) in obstacles:
        bl_x, bl_y = bottom_left
        width = top_right[0] - bl_x
        height = top_right[1] - bl_y
        ax.add_patch(Rectangle(bottom_left, width, height, 
                              facecolor='gray', alpha=0.5))

    # Plot start and goal points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    # Plot tree edges
    for edge in planner.edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 
                'b-', alpha=0.2)

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')

    ax.legend()
    plt.title('RRT Path Planning')
    plt.show()

def plan_shortest_path(start, goal, obstacles):
    """Main function to plan and display the path"""
    # Print initial information
    print("Grid Size: 20x20")
    print(f"Start Point: {start}")
    print(f"Goal Point: {goal}")
    print("Obstacles (bottom-left, top-right):")
    for obs in obstacles:
        print(f"  {obs}")

    # Initialize planner
    planner = RRTPlanner()
    
    # Plan the path
    success = planner.plan_path(start, goal, obstacles)
    
    # Get the path
    path = planner.reconstruct_path(start, goal)
    
    # Print results
    print("\nPlanning Result:")
    if success:
        print("Path found!")
    else:
        print("Could not reach goal. Showing path to closest point.")
    print("Path coordinates:")
    for point in path:
        print(f"  {point}")
    
    # Plot the result
    plot_path(start, goal, obstacles, path, planner)

# Example usage
if __name__ == "__main__":
    start = (2.0, 2.0)
    goal = (18.0, 18.0)
    obstacles = [((3, 3), (5, 5)),
                 ((8, 12), (10, 14)),
                 ((15, 2), (17, 4))]
    
    plan_shortest_path(start, goal, obstacles)