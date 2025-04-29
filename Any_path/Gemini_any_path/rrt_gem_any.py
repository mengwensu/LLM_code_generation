import matplotlib.pyplot as plt
import numpy as np
import random

def is_collision_free(point, obstacles):
  """
  Checks if a point is collision-free with any obstacles.

  Args:
      point: (x, y) coordinates of the point.
      obstacles: List of obstacles, where each obstacle is a tuple 
                  of (bottom_left, top_right) coordinates.

  Returns:
      True if the point is collision-free, False otherwise.
  """
  for obstacle in obstacles:
      x1, y1 = obstacle[0]
      x2, y2 = obstacle[1]
      if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
          return False
  return True

def sample_point():
  """
  Samples a random point within the grid.

  Returns:
    (x, y) coordinates of the sampled point.
  """
  return random.uniform(0, 20), random.uniform(0, 20)

def nearest_neighbor(point, tree):
  """
  Finds the nearest node in the tree to the given point.

  Args:
    point: (x, y) coordinates of the point.
      tree: List of nodes in the tree, where each node is a tuple 
          of (x, y) coordinates.

  Returns:
    (x, y) coordinates of the nearest node.
  """
  distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in tree]
  return tree[np.argmin(distances)]

def steer(from_node, to_node, step_size):
  """
  Steers towards the target point with a given step size.

  Args:
    from_node: (x, y) coordinates of the starting node.
      to_node: (x, y) coordinates of the target node.
      step_size: Distance to move towards the target.

  Returns:
    (x, y) coordinates of the new point.
  """
  direction = np.array(to_node) - np.array(from_node)
  direction = direction / np.linalg.norm(direction)
  new_point = from_node + step_size * direction
  return tuple(new_point)

def rrt(obstacles, start=(0, 0), goal=(20, 20), step_size=1, iterations=500):
  """
  Plans a path using the Rapidly-exploring Random Trees (RRT) algorithm.

  Args:
    obstacles: List of obstacles, where each obstacle is a tuple 
          of (bottom_left, top_right) coordinates.
    start: (x, y) coordinates of the starting point.
    goal: (x, y) coordinates of the goal point.
    step_size: Distance to move towards the target in each step.
    iterations: Number of iterations to run the RRT algorithm.

  Returns:
      List of nodes representing the planned path, or None if no path is found.
  """
  tree = [start]
  for _ in range(iterations):
      rand_point = sample_point()
      nearest_node = nearest_neighbor(rand_point, tree)
      new_point = steer(nearest_node, rand_point, step_size)
      if is_collision_free(new_point, obstacles):
          tree.append(new_point)

  # Check for connection to goal
  nearest_node = nearest_neighbor(goal, tree)
  while np.linalg.norm(np.array(nearest_node) - np.array(goal)) > step_size:
      new_point = steer(nearest_node, goal, step_size)
      if is_collision_free(new_point, obstacles):
          nearest_node = new_point
          tree.append(new_point)
      else:
          break

  # Reconstruct path if goal is reached
  if np.linalg.norm(np.array(nearest_node) - np.array(goal)) <= step_size:
      path = [goal]
      current_node = nearest_node
      while current_node != start:
          path.insert(0, current_node)
          current_node = nearest_neighbor(current_node, tree[:-1])  # Exclude goal from nearest neighbor search
      path.insert(0, start)
      return path
  else:
      return None

def plot_grid(grid, obstacles, path):
  """
  Plots the grid, obstacles, and path.

  Args:
    grid: 2D array representing the grid (not used in this implementation).
    obstacles: List of obstacles.
    path: List of nodes representing the planned path.
  """
  plt.figure(figsize=(10, 10))
  plt.xlim(0, 20)
  plt.ylim(0, 20)
  plt.grid(True)

  for obstacle in obstacles:
    x1, y1 = obstacle[0]
    x2, y2 = obstacle[1]
    plt.fill_between([x1, x2], [y1, y1], [y2, y2], color='black', alpha=0.5)

  if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'r-', linewidth=2)

  plt.plot(0, 0, 'go', markersize=8)
  plt.plot(20, 20, 'ro', markersize=8)

  plt.show()

def main():
  grid_size = 20
  obstacles = [
    ((3, 3), (5, 5)),
    ((10, 10), (15, 15)),
    ((1, 18), (5, 20)),
    ((15, 0), (20, 5))
  ]

  path = rrt(obstacles)

  if path:
    print("Path found:")
    print(path)
    plot_grid(None, obstacles, path)
  else:
    print("No path found.")

if __name__ == "__main__":
  main()
