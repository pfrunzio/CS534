from Board import Board
import pandas as pd

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
