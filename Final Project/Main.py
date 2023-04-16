import pandas as pd
import numpy as np
import sys

from Gridworld import Gridworld
from Gridworld import TileValue
from Gridworld import Action


def main(argv):

    command_format = "\nCommands: [gridworld file] [player mode (0 for off, 1 for on)]\n"
    print(command_format)

    argv = input("Enter arguments: ").split()
    if len(argv) != 2:
        print("Incorrect number of command line arguments")
        exit()

    player_mode = False
    if argv[1] == "1":
        player_mode = True

    gridworld = get_gridworld(argv[0])

    # Lets you play the game yourself, might be useful for debugging purposes
    if player_mode:
        _run_player_mode(gridworld)


def get_gridworld(file_name):
    gridworld = None

    try:
        directory = './Gridworlds/' + file_name
        board = pd.read_csv(directory, sep="\t", header=None)
        start_index = np.where(board == 'S')
        start = (start_index[0][0], start_index[1][0])

        board = board \
            .replace("S", TileValue.EMPTY) \
            .replace("-", TileValue.EMPTY) \
            .replace("W", TileValue.WATER) \
            .replace("M", TileValue.MOUNTAIN) \
            .replace("F", TileValue.FOOD) \
            .replace("B", TileValue.BOAR) \
            .values

        for row in range(len(board)):
            for col in range(len(board[0])):
                val = board[row][col]
                if isinstance(val, str) and val.isalpha():
                    board[row][col] = -1 * ord(val)

        gridworld = Gridworld(board.astype(int), start)
    except Exception as e:
        print("Exception while generating gridworld from file, ", e.__str__())

    return gridworld


def _run_player_mode(gridworld):
    while not gridworld.is_terminal:
        player_action = _get_player_action(gridworld)
        gridworld = gridworld.take_action(player_action)

    print("\nFinal world:\n")
    print(gridworld.__str__())


def _get_player_action(gridworld):

    print(gridworld.__str__())
    while True:
        player_action = input("Please enter an integer to select an action:\n" +
                              "1. Move up\n" +
                              "2. Move down\n" +
                              "3. Move left\n" +
                              "4. Move right\n" +
                              "5. Use tile\n" +
                              "6. Pick up item\n" +
                              "7. Use inventory\n")

        if player_action == "1":
            return Action.UP
        elif player_action == "2":
            return Action.DOWN
        elif player_action == "3":
            return Action.LEFT
        elif player_action == "4":
            return Action.RIGHT
        elif player_action == "5":
            return Action.USE_TILE
        elif player_action == "6":
            return Action.PICK_UP_ITEM
        elif player_action == "7":
            return Action.USE_INVENTORY
        else:
            print("Please enter a valid input\n")


if __name__ == "__main__":
    main(sys.argv[1:])
