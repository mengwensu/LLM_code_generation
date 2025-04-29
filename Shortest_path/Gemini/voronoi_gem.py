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

def plan_path_voronoi(grid, obstacles, start, end):
  """
  Plans the shortest path using the Voronoi algorithm.

  Args:
    grid: The grid with obstacles.
    obstacles: A list of obstacles.
    start: Starting point coordinates (x, y).
    end: Ending point coordinates (x, y).

  Returns:
    A list of points representing the planned path, or None if no path is found.
  """
  valid_points = get_valid_points(grid, obstacles)

  # Add start and end points to valid points
  valid_points.append(start)
  valid_points.append(end)

  # Create Voronoi diagram
  vor = Voronoi(np.array(valid_points))

  # Find the region containing the start point
  start_region = vor.point_region[valid_points.index(start)]

  # Perform region-based pathfinding (simplified)
  current_region = start_region
  path = [start]
  while current_region != vor.point_region[valid_points.index(end)]:
    # Find neighboring regions
    neighbors = vor.regions[current_region]
    neighboring_points = [valid_points[i] for i in neighbors if i != -1]

    # Choose the closest neighbor to the end point (heuristic)
    closest_neighbor = min(neighboring_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(end)))

    # Update current region and path
    current_region = vor.point_region[valid_points.index(closest_neighbor)]
    path.append(closest_neighbor)

    # Check if no path is found (stuck in a local minimum)
    if len(path) > len(valid_points) * 0.5:  # Arbitrary limit
      print("Warning: Pathfinding may be stuck. Returning incomplete path.")
      break

  return path

def plot_grid(grid, obstacles, path, start, end):
  """
  Plots the grid with obstacles, path, start, and end points.

  Args:
    grid: The grid with obstacles.
    obstacles: A list of obstacles.
    path: The planned path.
    start: Starting point coordinates.
    end: Ending point coordinates.
  """
  plt.figure(figsize=(10, 10))
  plt.imshow(grid, cmap='binary', origin='lower')
  plt.title("Grid with Obstacles and Path")
  plt.xlabel("X")
  plt.ylabel("Y")

  # Plot obstacles
  for obstacle in obstacles:
    x1, y1 = obstacle[0]
    x2, y2 = obstacle[1]
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r')

  # Plot path
  if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'g', linewidth=2)

  # Plot start and end points
  plt.plot(start[0], start[1], 'go', markersize=8)
  plt.plot(end[0], end[1], 'ro', markersize=8)

  plt.grid(True, which='both', linestyle='--')
  plt.xticks(np.arange(0, grid.shape[1] + 1, 1))
  plt.yticks(np.arange(0, grid.shape[0] + 1, 1))
  plt.show()

def main():
  grid_size = 20
  grid = create_grid(grid_size, grid_size)

  # Define obstacles (example)
  obstacles = [
      ((3, 3), (5, 5)),
      ((10, 10), (15, 12)),
      ((18, 2), (19, 5))
  ]

  grid = add_obstacles(grid, obstacles)

  start = (1, 1)
  end = (18, 18)

  path = plan_path_voronoi(grid, obstacles, start, end)

  if path:
    print("Planned Path:", path)
  else:
    print("No path found.")

  plot_grid(grid, obstacles, path, start, end)

if __name__ == "__main__":
  main()