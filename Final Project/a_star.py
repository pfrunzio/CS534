import heapq

from Gridworld import Action


def astar_search(start, end, grid):
    open_list = []
    heapq.heappush(open_list, (0, start))  # Priority queue for open list
    came_from = {}  # Dictionary to store the previous node in the optimal path
    g_score = {cell: float('inf') for row in grid for cell in row}  # g score for each cell
    g_score[start] = 0  # g score for the start cell
    f_score = {cell: float('inf') for row in grid for cell in row}  # f score for each cell
    f_score[start] = heuristic(start, end)  # f score for the start cell

    while open_list:
        _, current = heapq.heappop(open_list)  # Get the cell with the lowest f score from the open list

        if current == end:
            return reconstruct_path(came_from, end)  # If the goal is reached, return the path

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + grid[neighbor[0]][neighbor[1]]  # g score from current to neighbor
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # If no path is found

def heuristic(node, goal):
    """Manhattan distance heuristic."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def get_neighbors(node, grid):
    """Get valid neighbors of a node on the grid, accounting for walls."""
    neighbors = []
    row, col = node
    rows = len(grid)
    cols = len(grid[0])

    # Check top
    if row > 0 and grid[row - 1][col] >= 0:
        if grid[row - 1][col] != -5 and grid[row - 1][col] != 3:
            neighbors.append((row - 1, col))
    # Check bottom
    if row < rows - 1 and grid[row + 1][col] >= 0:
        if grid[row + 1][col] != -5 and grid[row + 1][col] != 3:
            neighbors.append((row + 1, col))
    # Check left
    if col > 0 and grid[row][col - 1] >= 0:
        if grid[row][col - 1] != -5 and grid[row][col - 1] != 3:
            neighbors.append((row, col - 1))
    # Check right
    if col < cols - 1 and grid[row][col + 1] >= 0:
        if grid[row][col + 1] != -5 and grid[row][col + 1] != 3:
            neighbors.append((row, col + 1))

    return neighbors


def reconstruct_path(came_from, end):
    """Reconstruct the optimal path from the came_from dictionary."""
    path = []
    while end in came_from:
        prev = came_from[end]
        if prev[0] < end[0]:
            path.append(Action.UP)
        elif prev[0] > end[0]:
            path.append(Action.DOWN)
        elif prev[1] < end[1]:
            path.append(Action.LEFT)
        else:
            path.append(Action.RIGHT)
        end = prev
    path.reverse()
    return path

