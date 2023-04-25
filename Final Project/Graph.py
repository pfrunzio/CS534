import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Genetic import Genetic
from GeneticDecision import GeneticDecision
from GeneticSequenceTable import GeneticSequenceTable
from GeneticSlice import GeneticSlice
from GeneticTable import GeneticTable
from Gridworld import Gridworld
from Gridworld import TileValue
from scipy.interpolate import interp1d
from RL import RL

from Gridworld import Action


def main():
    graph('Level1.txt', 1, 20)


def graph(file_name, level, time):
    gridworld = get_gridworld(file_name, level)

    g_data = Genetic(gridworld, time).run()

    sg_data = GeneticSlice(gridworld, time).run()
    # sg_data = []

    qt_data = RL(gridworld, time).start()
    # qt_data = []

    tg_data = GeneticTable(gridworld, time).run()
    # tg_data = []

    # stg_data = data.append(GeneticSequenceTable(gridworld).run())
    stg_data = []

    # dg_data = GeneticDecision(gridworld).run()
    dg_data = []

    title = f'Level {level} {file_name.split(".")[0]} World over {time} seconds'

    graph_comparison(g_data, sg_data, qt_data, tg_data, stg_data, dg_data, time, title)


def graph_comparison(g_data, sg_data, qt_data, tg_data, stg_data, dg_data, total_time, title):
    generation_num_index = 0
    max_score_index = 1
    average_score_index = 2
    time_index = 3

    graph_helper(g_data, sg_data, qt_data, tg_data, stg_data, dg_data, max_score_index, time_index, total_time, False, title)
    graph_helper(g_data, sg_data, qt_data, tg_data, stg_data, dg_data, max_score_index, time_index, total_time, True, title)
    # graph_helper(g_data, sg_data, qt_data, tg_data, stg_data, average_score_index, time_index, total_time)


def graph_helper(g_data, sg_data, qt_data, tg_data, stg_data, dg_data, y_index, x_index, total_time, interpolate, title):
    graph_data_helper(g_data, y_index, x_index, total_time, 'Genetic', interpolate)
    graph_data_helper(sg_data, y_index, x_index, total_time, 'Genetic-Slice', interpolate)
    graph_data_helper(qt_data, 1, 0, total_time, 'Q-Table', interpolate)
    graph_data_helper(tg_data, y_index, x_index, total_time, 'Genetic-Table', interpolate)
    graph_data_helper(stg_data, y_index, x_index, total_time, 'Genetic-Sequence-Table', interpolate)
    graph_data_helper(stg_data, y_index, x_index, total_time, 'Genetic-Sequence-Table', interpolate)
    graph_data_helper(dg_data, y_index, x_index, total_time, 'Genetic-Decision', interpolate)

    # add axis labels and a legend
    plt.xlabel('Time (s)')
    plt.ylabel('Turns Survived')
    plt.title(title)
    plt.legend()

    # show the plot
    plt.show()


def graph_data_helper(data, y_index, x_index, total_time, label, interpolate):
    if len(data) > 0:
        if interpolate is True:
            new_data = interpolate_data(data[y_index], data[x_index], total_time)
            plt.plot(new_data[0], new_data[1], label=label)
        else:
            plt.plot(data[x_index], data[y_index], label=label)


def interpolate_data(data, time, total_time):
    interp_func = interp1d(time, data, fill_value="extrapolate")
    new_time = np.arange(0, total_time, 1)
    new_data = interp_func(new_time)

    return [new_time, new_data]


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


if __name__ == "__main__":
    main()
