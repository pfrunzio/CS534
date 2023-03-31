import csv

import numpy as np
import pandas as pd
import os
import glob

from numpy import NaN


def main():
    # use glob to get all the csv files
    # in the folder
    path = "./4x4boards"
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    board_size = 4
    with open(f"ListOfBoards{board_size}x{board_size}.csv", "w", newline="") as csv_file:
        for f in csv_files:
            # read the csv file
            board_array = pd.read_csv(f, sep=',', header=None).replace('B', 0).replace(NaN, -1).values.astype(int)

            # print the location and filename
            size = len(board_array)
            cost = board_array[size - 1][0]
            board = np.delete(board_array, size - 1, 0)
            board_csv = np.reshape(board, (1, (size - 1) ** 2))[0]

            board_csv = np.append(board_csv, cost)
            writer = csv.writer(csv_file)
            writer.writerow(board_csv)
    # loop over the list of csv files


if __name__ == "__main__":
    main()
