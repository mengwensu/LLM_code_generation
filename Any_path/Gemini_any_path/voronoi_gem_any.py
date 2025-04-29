import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

def create_grid(width, height):
  """
  Creates an empty grid.

  Args:
    width: Width of the grid.
    height: Height of the grid.

  Returns:
    A 2D array representing the grid.
  """
  return np.zeros((height, width))

def add_obstacles(grid, obstacles):
  """
  Adds rectangular obstacles to the grid.

  Args:
    grid: The grid to add obstacles to.
    obstacles: A list of tuples, where each tuple represents an obstacle 
               and contains the coordinates of the bottom-left and top-right corners.

  Returns:
    The updated grid with obstacles.
  """
  for obstacle in obstacles:
    x1, y1 = obstacle[0]
    x2, y2 = obstacle[1]
    grid[y1:y2+1, x1:x2+1] = 1  # Mark obstacle cells as 1
  return grid

def get_valid_points(grid, obstacles):
  """
  Gets a list of valid points (points not inside obstacles) on the grid.

  Args:
    grid: The grid with obstacles.
    obstacles: A list of obstacles.

  Returns:
    A list of valid points as tuples (x, y).
  """
  valid_points = []
  for y in range(grid.shape[0]):
    for x in range(grid.shape[1]):
      if grid[y, x] == 0:
        valid_points.append((x, y))
  return valid_points

def plan_path_voronoi(start, goal, obstacles, grid_size=(20, 20)):
  """
  Plans the shortest path using the Voronoi algorithm.

  Args:
    start: Starting point (x, y) tuple.
    goal: Goal point (x, y) tuple.
    obstacles: A list of obstacles.
    grid_size: Size of the grid (width, height).

  Returns:
    A list of points representing the planned path, or None if the goal is not reachable.
  """
  grid = create_grid(*grid_size)
  grid = add_obstacles(grid, obstacles)
  valid_points = get_valid_points(grid, obstacles)

  # Add start and goal to valid points
  valid_points.append(start)
  valid_points.append(goal)

  # Create Voronoi diagram
  vor = Voronoi(np.array(valid_points))

  # Find the region containing the start point
  start_region = vor.point_region[valid_points.index(start)]

  # Perform region-based pathfinding
  current_region = start_region
  path = [start]
  while current_region != vor.point_region[valid_points.index(goal)]:
    neighbors = vor.regions[current_region]
    min_dist = float('inf')
    next_region = None

    for neighbor in neighbors:
      if neighbor == -1:  # Ignore unbounded regions
        continue
      neighbor_point = vor.vertices[neighbor]
      dist = np.linalg.norm(np.array(goal) - neighbor_point)
      if dist < min_dist:
        min_dist = dist
        next_region = neighbor

    if next_region is None:  # Goal not reachable
      return None

    path.append(tuple(vor.vertices[next_region]))
    current_region = next_region

  path.append(goal)
  return path

def plot_grid(grid, obstacles, path, start, goal):
  """
  Plots the grid, obstacles, and path.

  Args:
    grid: The grid.
    obstacles: A list of obstacles.
    path: The planned path.
    start: Starting point.
    goal: Goal point.
  """
  plt.figure(figsize=(8, 8))
  plt.imshow(grid, cmap='gray', origin='lower')
  plt.grid(which='both', linestyle='--')

  for obstacle in obstacles:
    x1, y1 = obstacle[0]
    x2, y2 = obstacle[1]
    plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color='red', alpha=0.5)

  if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'b-', linewidth=2)

  plt.scatter([start[0]], [start[1]], color='green', s=100, marker='o')
  plt.scatter([goal[0]], [goal[1]], color='red', s=100, marker='o')

  plt.xlim(0, grid.shape[1])
  plt.ylim(0, grid.shape[0])
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Voronoi Path Planning')
  plt.show()

if __name__ == "__main__":
  # Define obstacles
  obstacles = [
      ((3, 3), (5, 5)),
      ((10, 10), (12, 15)),
      ((15, 2), (18, 8))
  ]

  # Define start and goal
  start = (1, 1)
  goal = (18, 18)

  # Plan path
  path = plan_path_voronoi(start, goal, obstacles)

  # Print results
  print("Obstacles:", obstacles)
  print("Start:", start)
  print("Goal:", goal)
  if path:
    print("Path:", path)
  else:
    print("Goal not reachable.")

  # Plot the grid
  plot_grid(create_grid(20, 20), obstacles, path, start, goal)


