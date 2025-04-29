import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import math

class RRT:
    def __init__(self, start, goal, obstacles, grid_size=20, step_size=1.0, max_iter=1000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.parent = {tuple(self.start): None}
        self.safety_margin = 0.5

    def distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def is_in_obstacle(self, point):
        for (bottom_left, top_right) in self.obstacles:
            bl_x, bl_y = bottom_left
            tr_x, tr_y = top_right
            if (bl_x - self.safety_margin <= point[0] <= tr_x + self.safety_margin and
                bl_y - self.safety_margin <= point[1] <= tr_y + self.safety_margin):
                return True
        return False

    def is_collision_free(self, p1, p2):
        steps = int(self.distance(p1, p2) / 0.1) + 1
        for i in range(steps + 1):
            t = i / steps
            point = (1 - t) * p1 + t * p2
            if self.is_in_obstacle(point):
                return False
        return True

    def nearest_node(self, point):
        distances = [self.distance(node, point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, p_from, p_to):
        dist = self.distance(p_from, p_to)
        if dist <= self.step_size:
            return p_to
        theta = np.arctan2(p_to[1] - p_from[1], p_to[0] - p_from[0])
        return p_from + self.step_size * np.array([np.cos(theta), np.sin(theta)])

    def plan_path(self):
        for _ in range(self.max_iter):
            rand_point = np.array([random.uniform(0, self.grid_size),
                                 random.uniform(0, self.grid_size)])
            
            # Occasionally bias towards goal
            if random.random() < 0.1:
                rand_point = self.goal

            nearest = self.nearest_node(rand_point)
            new_point = self.steer(nearest, rand_point)

            if (0 <= new_point[0] <= self.grid_size and 
                0 <= new_point[1] <= self.grid_size and
                self.is_collision_free(nearest, new_point)):
                
                self.nodes.append(new_point)
                self.parent[tuple(new_point)] = nearest

                if self.distance(new_point, self.goal) < self.step_size:
                    self.nodes.append(self.goal)
                    self.parent[tuple(self.goal)] = new_point
                    return True

        return False

    def get_path(self):
        if tuple(self.goal) not in self.parent:
            # Return path to closest point to goal if goal not reached
            distances = [self.distance(node, self.goal) for node in self.nodes]
            closest = self.nodes[np.argmin(distances)]
            end_point = closest
        else:
            end_point = self.goal

        path = []
        current = end_point
        while current is not None:
            path.append(current)
            current = self.parent.get(tuple(current))
        return path[::-1]

def plot_path(start, goal, obstacles, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set grid and limits
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))

    # Plot obstacles
    for (bottom_left, top_right) in obstacles:
        bl_x, bl_y = bottom_left
        tr_x, tr_y = top_right
        width = tr_x - bl_x
        height = tr_y - bl_y
        ax.add_patch(Rectangle(bottom_left, width, height, 
                            facecolor='gray', alpha=0.5))

    # Plot path
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Path')

    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    ax.legend()
    ax.set_title('RRT Path Planning')
    plt.show()

def plan_path_rrt(start, goal, obstacles):
    # Validate inputs
    if not (0 <= start[0] <= 20 and 0 <= start[1] <= 20 and
            0 <= goal[0] <= 20 and 0 <= goal[1] <= 20):
        print("Start or goal coordinates out of bounds!")
        return

    # Print grid info
    print("Grid Size: 20x20")
    print("Start:", start)
    print("Goal:", goal)
    print("Obstacles:", obstacles)

    # Create and run RRT
    rrt = RRT(start, goal, obstacles)
    success = rrt.plan_path()
    path = rrt.get_path()

    print("\nPath found:", "Yes" if success else "No (showing closest path)")
    print("Path coordinates:", [tuple(p) for p in path])

    # Plot results
    plot_path(start, goal, obstacles, path)

# Example usage
if __name__ == "__main__":
    start = (2, 2)
    goal = (18, 18)
    obstacles = [((3, 3), (5, 5)), 
                 ((8, 12), (10, 14)),
                 ((14, 6), (16, 8))]
    
    plan_path_rrt(start, goal, obstacles)