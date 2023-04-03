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


def main(argv):
    command_format = "\ncommands: [Board Size NxN] [Number of Boards (You will get 2x this number of boards)] [Max Moves]"

    print(command_format)

    if len(argv) != 0 and argv[0] == "q":
        return

    if len(argv) < 3:
        print("Not enough command line arguments", command_format)
        return

    if len(argv) == 3:
        board_size = int(argv[0])
        number_of_boards = int(argv[1])
        max_moves = int(argv[2])
        generate_all_boards(board_size, number_of_boards, max_moves)
        return


def generate_all_boards(board_size, number_of_boards, max_moves):
    list_of_boards = []
    with open(f"ListOfBoards{board_size}x{board_size}.csv", "w", newline="") as f:
        for x in range(number_of_boards):
            front_board, back_board = generate_board_by_moves(board_size, max_moves)
            writer = csv.writer(f)

            if front_board is not None:
                writer.writerow(front_board)
            if back_board is not None:
                writer.writerow(back_board)

    return list_of_boards


def generate_board_by_moves(board_size, max_num_moves):
    board_area = board_size ** 2
    number_of_blanks = random.randint(2, board_size)

    num_moves = random.randint(1, max_num_moves)

    board_csv_front_blanks = random.sample(range(1, board_area + 1), board_area - number_of_blanks)
    board_csv_back_blanks = random.sample(range(1, board_area + 1), board_area - number_of_blanks)

    board_csv_back_blanks.sort()

    for x in range(0, number_of_blanks):
        board_csv_front_blanks.append(0)
        board_csv_back_blanks.append(0)

    board_csv_front_blanks.sort()

    board_csv_front_blanks = get_final_board(board_csv_front_blanks, board_size, num_moves)
    board_csv_back_blanks = get_final_board(board_csv_back_blanks, board_size, num_moves)

    return board_csv_front_blanks, board_csv_back_blanks


def get_final_board(board_csv, board_size, num_moves):

    board = Board(np.reshape(board_csv, (board_size, board_size)))
    board = make_random_moves(board, num_moves)

    try:
        back_cost = BoardGeneratorAStar.BoardGeneratorAStar(board, Algorithm.HEURISTIC_SLIDE, True).start()
    except Exception as e:
        return None

    board_csv = np.reshape(board.board, (1, board_size ** 2))
    board_csv = np.append(board_csv, back_cost)

    return board_csv


def make_random_moves(board, move_count):
    current_board = board

    for i in range(move_count):
        neighbors = current_board.neighbors()
        current_board = neighbors[random.randint(0, len(neighbors) - 1)]

    current_board.cost = 0

    return current_board


def extract_board_from_file(file):
    list_of_board = []
    list_of_board_csv = pd.read_csv(file, sep=',', header=None).values.astype(int)
    board_size = int((len(list_of_board_csv[0]) - 1) ** (1 / 2))
    for x in range(len(list_of_board_csv)):
        board_csv = list(list_of_board_csv[x])
        list_of_board.append(create_board_from_csv(board_csv, board_size))
    return list_of_board


def create_board_from_csv(board_csv, board_size):
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
