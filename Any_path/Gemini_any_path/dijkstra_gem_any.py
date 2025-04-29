import matplotlib.pyplot as plt
import numpy as np

def is_valid(x, y, grid_size, obstacles):
  """
  Checks if a given coordinate is within the grid bounds and not inside any obstacle.

  Args:
    x: x-coordinate.
    y: y-coordinate.
    grid_size: Size of the grid (tuple of (width, height)).
    obstacles: List of obstacle tuples, where each tuple contains 
               (bottom_left_x, bottom_left_y, top_right_x, top_right_y).

  Returns:
    True if the coordinate is valid, False otherwise.
  """
  if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
    return False
  for obstacle in obstacles:
    if obstacle[0] <= x <= obstacle[2] and obstacle[1] <= y <= obstacle[3]:
      return False
  return True

def get_neighbors(x, y, grid_size, obstacles):
  """
  Returns a list of valid neighbors for a given coordinate.

  Args:
    x: x-coordinate.
    y: y-coordinate.
    grid_size: Size of the grid (tuple of (width, height)).
    obstacles: List of obstacle tuples.

  Returns:
    List of valid neighbor coordinates (tuples of (x, y)).
  """
  neighbors = []
  for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
      if dx == 0 and dy == 0:
        continue
      neighbor_x = x + dx
      neighbor_y = y + dy
      if is_valid(neighbor_x, neighbor_y, grid_size, obstacles):
        neighbors.append((neighbor_x, neighbor_y))
  return neighbors

def dijkstra(start, goal, grid_size, obstacles):
  """
  Finds the shortest path using Dijkstra's algorithm.

  Args:
    start: Starting coordinates (tuple of (x, y)).
    goal: Goal coordinates (tuple of (x, y)).
    grid_size: Size of the grid (tuple of (width, height)).
    obstacles: List of obstacle tuples.

  Returns:
    A list of coordinates representing the shortest path, or None if no path exists.
  """
  open_set = {start}
  came_from = {}
  g_score = {start: 0}
  f_score = {start: heuristic(start, goal)}

  while open_set:
    current = min(open_set, key=lambda node: f_score[node])
    if current == goal:
      return reconstruct_path(came_from, current)

    open_set.remove(current)
    for neighbor in get_neighbors(current[0], current[1], grid_size, obstacles):
      tentative_g_score = g_score[current] + 1  # Cost to go from current to neighbor

      if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
        came_from[neighbor] = current
        g_score[neighbor] = tentative_g_score
        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
        if neighbor not in open_set:
          open_set.add(neighbor)

  return None

def reconstruct_path(came_from, current):
  """
  Reconstructs the path from the came_from dictionary.

  Args:
    came_from: Dictionary mapping nodes to their predecessors.
    current: The goal node.

  Returns:
    A list of coordinates representing the path.
  """
  total_path = [current]
  while current in came_from:
    current = came_from[current]
    total_path.insert(0, current)
  return total_path

def heuristic(a, b):
  """
  Manhattan distance heuristic.

  Args:
    a: Coordinates of the first point.
    b: Coordinates of the second point.

  Returns:
    Manhattan distance between the two points.
  """
  return abs(a[0] - b[0]) + abs(a[1] - b[1])

def plot_graph(grid_size, obstacles, start, goal, path):
  """
  Plots the grid, obstacles, start, goal, and path.

  Args:
    grid_size: Size of the grid (tuple of (width, height)).
    obstacles: List of obstacle tuples.
    start: Starting coordinates (tuple of (x, y)).
    goal: Goal coordinates (tuple of (x, y)).
    path: List of coordinates representing the path.
  """
  plt.figure(figsize=(8, 8))
  plt.xlim(0, grid_size[0])
  plt.ylim(0, grid_size[1])
  plt.grid(True)

  # Plot obstacles
  for obstacle in obstacles:
    plt.fill([obstacle[0], obstacle[2], obstacle[2], obstacle[0]], 
             [obstacle[1], obstacle[1], obstacle[3], obstacle[3]], 
             color='gray', alpha=0.5)

  # Plot start and goal
  plt.plot(start[0], start[1], 'go', markersize=10)
  plt.plot(goal[0], goal[1], 'ro', markersize=10)

  # Plot path
  if path:
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, 'b-')

  plt.show()

def plan_path(start, goal, grid_size, obstacles):
  """
  Plans the path using Dijkstra's algorithm.

  Args:
    start: Starting coordinates (tuple of (x, y)).
    goal: Goal coordinates (tuple of (x, y)).
    grid_size: Size of the grid (tuple of (width, height)).
    obstacles: List of obstacle tuples.

  Returns:
    A list of coordinates representing the shortest path, or None if no path exists.
  """
  path = dijkstra(start, goal, grid_size, obstacles)
  plot_graph(grid_size, obstacles, start, goal, path)
  return path

# Example usage
grid_size = (20, 20)
obstacles = [(3, 3, 5, 5), (10, 10, 12, 12), (7, 1, 9, 3)]
start = (0, 0)
goal = (19, 19)

path = plan_path(start, goal, grid_size, obstacles)
if path:
  print("Path found:")
  print(path)
else:
  print("No path found.")

