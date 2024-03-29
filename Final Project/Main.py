import pandas as pd
import numpy as np
import sys

from Genetic import Genetic
from GeneticDecision import GeneticDecision
from GeneticSequenceTable import GeneticSequenceTable
from GeneticSlice import GeneticSlice
from GeneticTable import GeneticTable
from Gridworld import Gridworld
from Gridworld import TileValue

from RL import RL

from Gridworld import Action

PLAYER_MODE = "player"
GENETIC_MODE = "genetic"
GENETIC_SLICE_MODE = "sgenetic"
Q_TABLE = "qtable"

GENETIC_TABLE_MODE = "tgenetic"
GENETIC_SEQUENCE_TABLE_MODE = "stgenetic"
GENETIC_DECISION = "dgenetic"

def main(argv):
    command_format = f'\nCommands: [gridworld file] [{PLAYER_MODE}/{GENETIC_MODE}/{GENETIC_SLICE_MODE}/{Q_TABLE}/{GENETIC_TABLE_MODE}/{GENETIC_SEQUENCE_TABLE_MODE}/{GENETIC_DECISION}] [level 1-4]\n'
    print(command_format)

    argv = input("Enter arguments: ").split()
    if len(argv) != 3:
        print("Incorrect number of command line arguments")
        exit()

    file_name = argv[0]
    mode = argv[1]
    level = argv[2]

    gridworld = get_gridworld(file_name, level)


    time = 100
    
    # Lets you play the game yourself, might be useful for debugging purposes
    if mode == PLAYER_MODE:
        _run_player_mode(gridworld)
    elif mode == GENETIC_MODE:
        Genetic(gridworld, time).run()
    elif mode == GENETIC_SLICE_MODE:
        GeneticSlice(gridworld, time).run()
    elif mode == Q_TABLE:
        RL(gridworld, time).start()
    elif mode == GENETIC_TABLE_MODE:
        GeneticTable(gridworld, time).run()
    elif mode == GENETIC_SEQUENCE_TABLE_MODE:
        GeneticSequenceTable(gridworld).run()
    elif mode == GENETIC_DECISION:
        GeneticDecision(gridworld).run()



def get_gridworld(file_name, level):
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

        gridworld = Gridworld(board.astype(int), start, level)
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
