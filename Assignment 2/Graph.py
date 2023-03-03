import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Gridworld import Gridworld, Value
from RL import RL


def main():

    def get_gridworld(filename):
        directory = './Gridworlds/' + filename
        board = pd.read_csv(directory, sep="	", header=None)
        start = (np.where(board == 'S')[0][0], np.where(board == 'S')[1][0])
        board = board \
            .replace('+', Value.COOKIE) \
            .replace("-", Value.GLASS) \
            .replace("X", Value.BARRIER) \
            .replace("S", Value.EMPTY) \
            .values.astype(int)

        return Gridworld(board, start)

    def graph_mean_reward(mean_rewards):

        x, y = zip(*mean_rewards)

        plt.scatter(x, y)
        plt.ylabel("Mean Reward")
        plt.xlabel("Time")
        plt.title(f'Mean Reward vs Time')
        plt.show()

    def graph_epsilons_mean_reward():
        intermediate_gridworld1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2 = get_gridworld("intermediate.txt")
        intermediate_gridworld3 = get_gridworld("intermediate.txt")
        intermediate_gridworld4 = get_gridworld("intermediate.txt")

        driver1 = RL(intermediate_gridworld1, 5, -0.04, .7, False)
        driver2 = RL(intermediate_gridworld2, 5, -0.04, .7, False)
        driver3 = RL(intermediate_gridworld3, 5, -0.04, .7, False)
        driver4 = RL(intermediate_gridworld4, 5, -0.04, .7, False)

        epsilon1 = driver1.graph_start(.01, .99, False)
        epsilon2 = driver2.graph_start(.1, .99, False)
        epsilon3 = driver3.graph_start(.3, .99, False)
        exploration = driver4.start()

        x1, y1 = zip(*epsilon1)
        x2, y2 = zip(*epsilon2)
        x3, y3 = zip(*epsilon3)
        x4, y4 = zip(*exploration)

        plt.scatter(x1, y1, c='b', marker='x', label='epsilon=0.01')
        plt.scatter(x2, y2, c='r', marker='o', label='epsilon=0.1')
        plt.scatter(x3, y3, c='g', marker='^', label='epsilon=0.3')
        plt.scatter(x4, y4, label='our method')

        plt.legend(loc='upper left')
        plt.ylabel("Mean Reward")
        plt.xlabel("Time")
        plt.title(f'Mean Reward vs Time')
        plt.show()

    def graph_initial_epsilon():
        intermediate_gridworld1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2 = get_gridworld("intermediate.txt")
        intermediate_gridworld3 = get_gridworld("intermediate.txt")
        intermediate_gridworld4 = get_gridworld("intermediate.txt")
        intermediate_gridworld5 = get_gridworld("intermediate.txt")

        driver1 = RL(intermediate_gridworld1, 5, -0.04, .7, False)
        driver2 = RL(intermediate_gridworld2, 5, -0.04, .7, False)
        driver3 = RL(intermediate_gridworld3, 5, -0.04, .7, False)
        driver4 = RL(intermediate_gridworld4, 5, -0.04, .7, False)
        driver5 = RL(intermediate_gridworld5, 5, -0.04, .7, False)

        epsilon1 = driver1.graph_start(.1, .99, True)
        epsilon2 = driver2.graph_start(.3, .99, True)
        epsilon3 = driver3.graph_start(.5, .99, True)
        epsilon4 = driver4.graph_start(.7, .99, True)
        epsilon5 = driver5.graph_start(1, .99, True)

        x1, y1 = zip(*epsilon1)
        x2, y2 = zip(*epsilon2)
        x3, y3 = zip(*epsilon3)
        x4, y4 = zip(*epsilon4)
        x5, y5 = zip(*epsilon5)

        plt.scatter(x1, y1, c='b', marker='x', label='init_epsilon=0.1')
        plt.scatter(x2, y2, c='r', marker='o', label='init_epsilon=0.3')
        plt.scatter(x3, y3, c='g', marker='^', label='init_epsilon=0.5')
        plt.scatter(x4, y4, c='k', marker='o', label='init_epsilon=0.7')
        plt.scatter(x5, y5, marker='x', label='init_epsilon=1')

        plt.legend(loc='upper left')
        plt.ylabel("Mean Reward")
        plt.xlabel("Time")
        plt.title(f'Mean Reward vs Time (Decay Rate .99)')
        plt.show()

    def graph_part3_vs_part4():
        intermediate_gridworld1_1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2_1 = get_gridworld("intermediate.txt")

        no_time = RL(intermediate_gridworld1_1, 10, -0.04, .7, False)
        time = RL(intermediate_gridworld2_1, 10, -0.04, .7, True)

        no_time_data = no_time.start()
        time_data = time.start()

        no_time_x, no_time_y = zip(*no_time_data)
        time_x, time_y = zip(*time_data)

        plt.scatter(no_time_x, no_time_y, c='b', marker='x', label='No Time')
        plt.scatter(time_x, time_y, c='r', marker='o', label='Time')

        plt.legend(loc='upper left')
        plt.ylabel("Mean Reward")
        plt.xlabel("Time")
        plt.title(f'Mean Reward vs Time')
        plt.show()

    graph_part3_vs_part4()
    # graph_epsilons_mean_reward()
    # graph_initial_epsilon()

if __name__ == "__main__":
    main()
