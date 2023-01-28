import pandas as pd
import sys
import HillClimbing
import AStar

COMMAND_ARGUMENT_ASTAR = "npuzzle"
COMMAND_ARGUMENT_HILLCLIMB = "greedy"


def main(argv):
    if len(argv) <= 0:
        argv = get_input()

    if len(argv) < 3:
        raise Exception("Not enough command line arguments")

    board = None

    try:
        directory = './Boards/' + argv[1]
        board = pd.read_csv(directory, sep=',', header=None).replace('B', 0).values.astype(int)
    except Exception as e:
        print(e)
        return

    algorithm = argv[0]

    driver = None

    if algorithm == COMMAND_ARGUMENT_HILLCLIMB:

        try:
            seconds = int(argv[2])
        except:
            print("Exception: Invalid seconds for command argument 3")
            return
        driver = HillClimbing.HillClimbing(board, seconds)

    elif algorithm == COMMAND_ARGUMENT_ASTAR:

        heuristic = argv[2]
        bool = argv[3].lower()

        weight = None

        if bool == "true":
            weight = True
        elif bool == "false":
            weight = False

        print(heuristic)

        if not heuristic.lower() in [AStar.HEURISTIC_TELEPORT, AStar.HEURISTIC_SLIDE]:
            raise Exception("Unknown heuristic for command argument 2")

        driver = AStar.AStar(board, heuristic, weight)
    else:
        raise Exception("Unknown algorithm for command argument 1")

    driver.start()


def get_input():
    user_input = input("Enter arguments:")
    return user_input.split()


if __name__ == "__main__":
    main(sys.argv[1:])
