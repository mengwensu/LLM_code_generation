import matplotlib.pyplot as plt
import numpy as np
import heapq
def create_grid(width, height):
  """
  Creates an empty grid.

  Args:
    width: Width of the grid.
    height: Height of the grid.

  Returns:
    A 2D list representing the grid, initialized with 0s.
  """
  return [[0] * width for _ in range(height)]

def add_obstacles(grid, obstacles):
  """
  Adds rectangular obstacles to the grid.

  Args:
    grid: The grid to modify.
    obstacles: A list of tuples, where each tuple represents an obstacle 
               and contains the coordinates of the bottom-left and top-right corners.

  Modifies the grid by marking obstacle cells with 1.
  """
  for obstacle in obstacles:
    (x1, y1), (x2, y2) = obstacle
    for x in range(x1, x2 + 1):
      for y in range(y1, y2 + 1):
        grid[y][x] = 1

def get_neighbors(node, grid):
  """
  Gets the valid neighbors of a node.

  Args:
    node: The current node (x, y) coordinates.
    grid: The grid.

  Returns:
    A list of valid neighbor nodes.
  """
  neighbors = []
  x, y = node
  rows, cols = len(grid), len(grid[0])

  for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
      if dx == 0 and dy == 0:
        continue  # Skip the current node
      nx, ny = x + dx, y + dy
      if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
        neighbors.append((nx, ny))

  return neighbors

def calculate_distance(node1, node2):
  """
  Calculates the Euclidean distance between two nodes.

  Args:
    node1: The first node (x, y) coordinates.
    node2: The second node (x, y) coordinates.

  Returns:
    The Euclidean distance between the two nodes.
  """
  x1, y1 = node1
  x2, y2 = node2
  return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def dijkstra(grid, start, end):
  """
  Finds the shortest path using Dijkstra's algorithm.

  Args:
    grid: The grid.
    start: The starting node (x, y) coordinates.
    end: The ending node (x, y) coordinates.

  Returns:
    A list of nodes representing the shortest path, or None if no path is found.
  """
  rows, cols = len(grid), len(grid[0])
  dist = [[float('inf')] * cols for _ in range(rows)]
  prev = [[None] * cols for _ in range(rows)]
  dist[start[1]][start[0]] = 0
  visited = set()
  queue = [(0, start)]

  while queue:
    _, current = heapq.heappop(queue)
    if current == end:
      path = []
      while current:
        path.insert(0, current)
        current = prev[current[1]][current[0]]
      return path

    if current in visited:
      continue

    visited.add(current)

    for neighbor in get_neighbors(current, grid):
      alt = dist[current[1]][current[0]] + calculate_distance(current, neighbor)
      if alt < dist[neighbor[1]][neighbor[0]]:
        dist[neighbor[1]][neighbor[0]] = alt
        prev[neighbor[1]][neighbor[0]] = current
        heapq.heappush(queue, (alt, neighbor))

  return None  # No path found

def plot_grid(grid, obstacles, path, start, end):
  """
  Plots the grid with obstacles and the planned path.

  Args:
    grid: The grid.
    obstacles: A list of obstacles.
    path: The planned path.
    start: The starting node.
    end: The ending node.
  """
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_xlim(0, len(grid[0]))
  ax.set_ylim(0, len(grid))
  ax.set_xticks(range(len(grid[0])))
  ax.set_yticks(range(len(grid)))
  ax.grid(True)

  # Plot obstacles
  for obstacle in obstacles:
    (x1, y1), (x2, y2) = obstacle
    obstacle_x = [x1 - 0.5, x2 + 0.5, x2 + 0.5, x1 - 0.5, x1 - 0.5]
    obstacle_y = [y1 - 0.5, y1 - 0.5, y2 + 0.5, y2 + 0.5, y1 - 0.5]
    ax.fill(obstacle_x, obstacle_y, color='gray')

  # Plot path
  if path:
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, color='blue', marker='o')

  # Plot start and end points
  ax.plot(start[0], start[1], 'go')
  ax.plot(end[0], end[1], 'ro')

  plt.show()

def main():
  grid_size = 20
  grid = create_grid(grid_size, grid_size)

  # Example obstacles
  obstacles = [
      ((3, 3), (5, 5)),
      ((10, 10), (12, 15)),
      ((17, 5), (19, 8))
  ]
  add_obstacles(grid, obstacles)

  start = (1, 1)
  end = (18, 18)

  path = dijkstra(grid, start, end)

  if path:
    print("Path found:")
    print(path)
  else:
    print("No path found.")

  plot_grid(grid, obstacles, path, start, end)

if __name__ == "__main__":
  main()