from enum import Enum
import heapq

from Gridworld import Action

# Define the possible movements from a cell to its neighbors
MOVEMENTS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

def a_star(grid, start, goal):
    # Get the dimensions of the grid
    rows = len(grid)
    cols = len(grid[0])

    # Initialize the open and closed sets
    open_set = []
    heapq.heappush(open_set, (0, start)) # Use a priority queue to store cells with their f-cost
    came_from = {}
    g_score = {(row, col): float('inf') for row in range(rows) for col in range(cols)} # Initialize g_score for all cells
    g_score[start] = 0
    f_score = {(row, col): float('inf') for row in range(rows) for col in range(cols)} # Initialize f_score for all cells
    f_score[start] = heuristic(start, goal)

    # Start the A* search
    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Goal reached, reconstruct the path and return it as a list of movements
            path = []
            while current in came_from:
                prev = came_from[current]
                move = get_move(prev, current)
                path.append(move)
                current = prev
            path.reverse()
            return path

        for move in MOVEMENTS:
            row, col = current
            new_row = row + move.value[0]
            new_col = col + move.value[1]

            if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row][new_col] not in [-5, 3]:
                # Check if the new cell is within the grid and not blocked by an obstacle
                tentative_g_score = g_score[current] + 1  # Assumes constant cost for moving to a neighbor

                if tentative_g_score < g_score[(new_row, new_col)]:
                    # Update the g-score and f-score of the neighbor
                    came_from[(new_row, new_col)] = current
                    g_score[(new_row, new_col)] = tentative_g_score
                    f_score[(new_row, new_col)] = tentative_g_score + heuristic((new_row, new_col), goal)
                    heapq.heappush(open_set, (f_score[(new_row, new_col)], (new_row, new_col)))

    # No path found
    return None

def heuristic(cell, goal):
    # Manhattan distance heuristic
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def get_move(prev, current):
    # Get the Action enum representing the move from the previous cell to the current cell
    move = (current[0] - prev[0], current[1] - prev[1])
    return Action(move[0], move[1])
