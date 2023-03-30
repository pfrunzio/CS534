import random
import sys
import threading
import csv
import numpy as np
import pandas as pd

import AStar
import Algorithm
import BoardGeneratorAStar
from Board import Board


def main1(argv):
    command_format = "\ncommands: [Board Size NxN] [Number of Boards]"

    print(command_format)

    while True:
        argv = get_input()

        if len(argv) != 0 and argv[0] == "q":
            return

        if len(argv) < 2:
            print("Not enough command line arguments", command_format)
            continue

        if len(argv) == 2:
            break

    board_size = int(argv[0])
    number_of_boards = int(argv[1])


    with open(f"ListOfBoards{board_size}x{board_size}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(generate_all_boards(board_size, number_of_boards))


def generate_all_boards(board_size, number_of_boards):
    list_of_boards = []
    for x in range(number_of_boards):
        board = generate_board(board_size)
        if board is not None:
            list_of_boards.append(board)
    return list_of_boards


def generate_board(board_size):
    board_area = board_size**2
    number_of_blanks = random.randint(2, round(board_area * 0.4))

    board_csv = random.sample(range(1, board_area + 1), board_area - number_of_blanks)

    for x in range(0, number_of_blanks):
        board_csv.append(0)

    random.shuffle(board_csv)

    board = np.reshape(board_csv, (board_size, board_size))

    try:
        cost = BoardGeneratorAStar.BoardGeneratorAStar(Board(board), Algorithm.HEURISTIC_SLIDE, True).start()
    except Exception as e:
        return

    board_csv.append(cost)

    return board_csv

def extract_board_from_file(file):
    list_of_board = []
    list_of_board_csv = pd.read_csv(file, sep=',', header=None).values.astype(int)
    board_size = int((len(list_of_board_csv[0])-1)**(1/2))
    for x in range(len(list_of_board_csv)):
        board_csv = list(list_of_board_csv[x])
        list_of_board.append(create_board_from_csv(board_csv, board_size))
    return list_of_board


def create_board_from_csv(board_csv,board_size):
    cost = board_csv.pop()
    board = Board(reformat_board_csv(board_csv, board_size), cost=cost)
    return board

def reformat_board_csv(board_csv, board_size):
    return np.reshape(board_csv, (board_size, board_size))

def get_input():
    user_input = input("Enter arguments:")
    return user_input.split()


if __name__ == "__main__":
    main(sys.argv[1:])
