import pandas as pd

from Board import Board
from HillClimbing import HillClimbing


def main():
    def _get_board(board_name):
        directory = './Boards/' + board_name
        board = Board(pd.read_csv(directory, sep=',', header=None).replace('B', 0).values.astype(int))

        return board

    seconds = 6
    board1 = _get_board("board1.csv")
    board2 = _get_board("board2.csv")
    # HillClimbing(board, False, seconds).graph_greedy_vs_annealing_vs_time(board, '(3x3 Board)')
    HillClimbing(board1, False, seconds).graph_annealing_temp_vs_time(board1, f'({len(board1.board)}x{len(board1.board)} Board)')
    HillClimbing(board2, False, seconds).graph_annealing_temp_vs_time(board2, f'({len(board2.board)}x{len(board2.board)} Board)')

    HillClimbing(board1, False, seconds).graph_annealing_temp_vs_time(board1, f'({len(board1.board)}x{len(board1.board)} Board) (Time Modification)')
    HillClimbing(board2, False, seconds).graph_annealing_temp_vs_time(board2, f'({len(board2.board)}x{len(board2.board)} Board) (Time Modification)')




if __name__ == "__main__":
    main()



