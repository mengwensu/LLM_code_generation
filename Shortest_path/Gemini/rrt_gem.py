import matplotlib.pyplot as plt
import numpy as np
import random

def generate_grid(width, height):
  """
  Generates an empty grid.

  Args:
    width: Width of the grid.
    height: Height of the grid.

  Returns:
    A 2D numpy array representing the grid.
  """
  return np.zeros((height, width))

def draw_grid(grid, obstacles, start, end, path):
  """
  Draws the grid with obstacles, start, end, and path.

  Args:
    grid: The grid to draw.
    obstacles: List of obstacles.
    start: Starting point coordinates.
    end: Ending point coordinates.
    path: List of points in the path.
  """
  plt.figure(figsize=(8, 8))
  plt.imshow(grid, cmap='gray', origin='lower')
  plt.grid(which='both', linestyle='--')
  plt.xticks(range(grid.shape[1]))
  plt.yticks(range(grid.shape[0]))

  # Draw obstacles
  for obstacle in obstacles:
    x1, y1, x2, y2 = obstacle  # Unpack the obstacle tuple correctly
    plt.fill_between([x1, x2], [y1, y2], [y1, y2], color='black')

  # Draw start and end points
  plt.scatter(start[0], start[1], color='green', s=100)
  plt.scatter(end[0], end[1], color='red', s=100)

  # Draw path
  if path:
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, color='blue')

  plt.show()

def is_valid_point(grid, point, obstacles, margin=0.5):
  """
  Checks if a point is valid (not within an obstacle).

  Args:
    grid: The grid.
    point: The point to check (x, y).
    obstacles: List of obstacles.
    margin: Safety margin around obstacles.

  Returns:
    True if the point is valid, False otherwise.
  """
  x, y = point
  if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
    return False

  for obstacle in obstacles:
    x1, y1, x2, y2 = obstacle  # Unpack the obstacle tuple correctly
    if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
      return False

  return True

def nearest_neighbor(points, new_point):
  """
  Finds the nearest point in a list of points.

  Args:
    points: List of points.
    new_point: The new point to compare to.

  Returns:
    The nearest point in the list.
  """
  distances = [np.linalg.norm(np.array(p) - np.array(new_point)) for p in points]
  nearest_index = np.argmin(distances)
  return points[nearest_index]

def steer(from_point, to_point, step_size):
  """
  Steers towards a point with a given step size.

  Args:
    from_point: Starting point.
    to_point: Target point.
    step_size: Maximum distance to move.

  Returns:
    The new point after steering.
  """
  direction = np.array(to_point) - np.array(from_point)
  distance = np.linalg.norm(direction)
  if distance <= step_size:
    return to_point
  else:
    direction = direction / distance * step_size
    return tuple(np.array(from_point) + direction)

def rrt(grid, start, end, obstacles, step_size=1, max_iterations=1000):
  """
  RRT algorithm to find a path.

  Args:
    grid: The grid.
    start: Starting point.
    end: Ending point.
    obstacles: List of obstacles.
    step_size: Step size for each iteration.
    max_iterations: Maximum number of iterations.

  Returns:
    A list of points representing the path, or None if no path is found.
  """
  path = [start]
  for _ in range(max_iterations):
    # Sample a random point
    rand_point = (random.uniform(0, grid.shape[1]), random.uniform(0, grid.shape[0]))

    # Find nearest neighbor
    nearest_node = nearest_neighbor(path, rand_point)

    # Steer towards the random point
    new_point = steer(nearest_node, rand_point, step_size)

    # Check if the new point is valid
    if is_valid_point(grid, new_point, obstacles):
      path.append(new_point)

      # Check if the goal is reached
      if np.linalg.norm(np.array(new_point) - np.array(end)) < step_size:
        path.append(end)
        return path

  return path

if __name__ == "__main__":
  # Define grid dimensions
  grid_width = 20
  grid_height = 20

  # Define obstacles (example)
  obstacles = [
      ((3, 3), (5, 5)),  # Correctly define obstacles as tuples of tuples
      ((10, 10), (12, 15)),
      ((15, 2), (18, 8))
  ]

  # Define start and end points
  start = (1, 1)
  end = (18, 18)

  # Create empty grid
  grid = generate_grid(grid_width, grid_height)

  # Find path using RRT
  path = rrt(grid, start, end, obstacles)

  # Draw the grid and path
  draw_grid(grid, obstacles, start, end, path)