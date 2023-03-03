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

    def graph_epsilons_mean_reward(time):
        intermediate_gridworld1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2 = get_gridworld("intermediate.txt")
        intermediate_gridworld3 = get_gridworld("intermediate.txt")
        intermediate_gridworld4 = get_gridworld("intermediate.txt")

        time

        driver1 = RL(intermediate_gridworld1, time, -0.04, .7, False)
        driver2 = RL(intermediate_gridworld2, time, -0.04, .7, False)
        driver3 = RL(intermediate_gridworld3, time, -0.04, .7, False)
        driver4 = RL(intermediate_gridworld4, time, -0.04, .7, False)

        epsilon1, epsilons1 = driver1.graph_start(.01, .99, False)
        epsilon2, epsilons2 = driver2.graph_start(.1, .99, False)
        epsilon3, epsilons3 = driver3.graph_start(.3, .99, False)
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

    def graph_initial_epsilon(time, decay_rate):
        intermediate_gridworld1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2 = get_gridworld("intermediate.txt")
        intermediate_gridworld3 = get_gridworld("intermediate.txt")
        intermediate_gridworld4 = get_gridworld("intermediate.txt")
        intermediate_gridworld5 = get_gridworld("intermediate.txt")

        driver1 = RL(intermediate_gridworld1, time, -0.04, .7, False)
        driver2 = RL(intermediate_gridworld2, time, -0.04, .7, False)
        driver3 = RL(intermediate_gridworld3, time, -0.04, .7, False)
        driver4 = RL(intermediate_gridworld4, time, -0.04, .7, False)
        driver5 = RL(intermediate_gridworld5, time, -0.04, .7, False)

        epsilon1, epsilons1 = driver1.graph_start(.1, decay_rate, True)
        epsilon2, epsilons2 = driver2.graph_start(.3, decay_rate, True)
        epsilon3, epsilons3 = driver3.graph_start(.5, decay_rate, True)
        epsilon4, epsilons4 = driver4.graph_start(.7, decay_rate, True)
        epsilon5, epsilons5 = driver5.graph_start(1, decay_rate, True)

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
        plt.title(f'Mean Reward vs Time (Decay Rate {decay_rate})')
        plt.show()

    def graph_decay_rate(time, epsilon):
        intermediate_gridworld1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2 = get_gridworld("intermediate.txt")
        intermediate_gridworld3 = get_gridworld("intermediate.txt")
        intermediate_gridworld4 = get_gridworld("intermediate.txt")
        intermediate_gridworld5 = get_gridworld("intermediate.txt")

        driver1 = RL(intermediate_gridworld1, time, -0.04, .7, False)
        driver2 = RL(intermediate_gridworld2, time, -0.04, .7, False)
        driver3 = RL(intermediate_gridworld3, time, -0.04, .7, False)
        driver4 = RL(intermediate_gridworld4, time, -0.04, .7, False)
        driver5 = RL(intermediate_gridworld5, time, -0.04, .7, False)

        decay1, epsilons1 = driver1.graph_start(epsilon, .999, True)
        decay2, epsilons2 = driver2.graph_start(epsilon, .99, True)
        decay3, epsilons3 = driver3.graph_start(epsilon, .9, True)
        decay4, epsilons4 = driver4.graph_start(epsilon, .75, True)
        decay5, epsilons5 = driver5.graph_start(epsilon, .5, True)

        x1, y1 = zip(*decay1)
        x2, y2 = zip(*decay2)
        x3, y3 = zip(*decay3)
        x4, y4 = zip(*decay4)
        x5, y5 = zip(*decay5)

        plt.scatter(x1, y1, c='b', marker='x', label='decay_rate=0.999')
        plt.scatter(x2, y2, c='r', marker='o', label='decay_rate=0.99')
        plt.scatter(x3, y3, c='g', marker='^', label='decay_rate=0.9')
        plt.scatter(x4, y4, c='k', marker='o', label='decay_rate=0.75')
        plt.scatter(x5, y5, marker='x', label='decay_rate=0.5')

        plt.legend(loc='upper left')
        plt.ylabel("Mean Reward")
        plt.xlabel("Time")
        plt.title(f'Mean Reward vs Time (Epsilon {epsilon})')
        plt.show()

    def graph_part3_vs_part4(time):
        intermediate_gridworld1_1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2_1 = get_gridworld("intermediate.txt")

        no_time = RL(intermediate_gridworld1_1, time, -0.04, .7, False)
        time = RL(intermediate_gridworld2_1, time, -0.04, .7, True)

        no_time_data, epsilons1 = no_time.start()
        time_data, epsilons2 = time.start()

        no_time_x, no_time_y = zip(*no_time_data)
        time_x, time_y = zip(*time_data)

        plt.scatter(no_time_x, no_time_y, c='b', marker='x', label='No Time')
        plt.scatter(time_x, time_y, c='r', marker='o', label='Time')

        plt.legend(loc='upper left')
        plt.ylabel("Mean Reward")
        plt.xlabel("Time")
        plt.title(f'Mean Reward vs Time')
        plt.show()

    def graph_epsilon_vs_time(time):
        intermediate_gridworld1_1 = get_gridworld("intermediate.txt")
        intermediate_gridworld2_1 = get_gridworld("intermediate.txt")

        no_time = RL(intermediate_gridworld1_1, time, -0.04, .7, False)
        time = RL(intermediate_gridworld2_1, time, -0.04, .7, True)

        no_time_data, epsilons1 = no_time.start()
        time_data, epsilons2 = time.start()

        x1, y1 = zip(*epsilons1)
        x2, y2 = zip(*epsilons2)

        plt.scatter(x1, y1, c='b', marker='x', label='No Time')
        plt.scatter(x2, y2, c='r', marker='o', label='Time')

        plt.legend(loc='upper left')
        plt.ylabel("Epsilon")
        plt.xlabel("Time")
        plt.title(f'Epsilon vs Time')
        plt.show()

    # graph_initial_epsilon(5, .99)
    # graph_decay_rate(5, 1)
    # graph_part3_vs_part4(5)
    graph_epsilon_vs_time(5)
    # graph_epsilons_mean_reward(5)


if __name__ == "__main__":
    main()
