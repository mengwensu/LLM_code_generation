import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import KDTree

class RRT:
    def __init__(self, start, goal, obstacles, step_size=1.5, max_iter=500):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [start]
        self.parent = {start: None}

    def is_collision_free(self, point):
        x, y = point
        for (x1, y1), (x2, y2) in self.obstacles:
            if x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5:
                return False
        return 0 <= x <= 20 and 0 <= y <= 20

    def get_nearest(self, point):
        tree = KDTree(self.nodes)
        _, idx = tree.query(point)
        return self.nodes[idx]

    def steer(self, from_point, to_point):
        vector = np.array(to_point) - np.array(from_point)
        dist = np.linalg.norm(vector)
        if dist < self.step_size:
            return to_point if self.is_collision_free(to_point) else from_point
        new_point = tuple(np.array(from_point) + self.step_size * (vector / dist))
        return new_point if self.is_collision_free(new_point) else from_point

    def plan(self):
        for _ in range(self.max_iter):
            rand_point = (random.uniform(0, 20), random.uniform(0, 20))
            nearest = self.get_nearest(rand_point)
            new_point = self.steer(nearest, rand_point)
            
            if new_point not in self.nodes and self.is_collision_free(new_point):
                self.nodes.append(new_point)
                self.parent[new_point] = nearest
                if np.linalg.norm(np.array(new_point) - np.array(self.goal)) < self.step_size:
                    self.nodes.append(self.goal)
                    self.parent[self.goal] = new_point
                    return self.extract_path()
        return self.extract_path()

    def extract_path(self):
        path = []
        node = self.goal if self.goal in self.parent else max(self.nodes, key=lambda n: np.linalg.norm(np.array(n) - np.array(self.goal)))
        while node is not None:
            path.append(node)
            node = self.parent[node]
        return path[::-1]

    def plot(self, path):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks(np.arange(0, 21, 1))
        ax.set_yticks(np.arange(0, 21, 1))
        ax.grid(True)
        
        for (x1, y1), (x2, y2) in self.obstacles:
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color='black'))
        
        ax.scatter(*self.start, color='green', s=100, label='Start')
        ax.scatter(*self.goal, color='red', s=100, label='Goal')
        
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='blue', linewidth=2)
        
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.legend()
        plt.show()

# Example usage
obstacles = [((3, 3), (5, 5)), ((10, 10), (12, 12)), ((7, 15), (10, 18))]
start = (1, 1)
goal = (18, 18)
rrt = RRT(start, goal, obstacles)
path = rrt.plan()
rrt.plot(path)
