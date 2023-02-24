import pandas as pd
import sys
import numpy as np

from Gridworld import Gridworld
from Gridworld import Value
from RL import RL


def main(argv):

    command_format = "\ncommands: [gridworld file] [seconds to run for] [per action reward] [transition model] [time based?]\n"

    print(command_format)

    while True:
        argv = get_input()

        if len(argv) != 0 and argv[0] == "q":
            return

        if len(argv) < 5:
            print("Not enough command line arguments", command_format)
            continue

        gridworld = None
        
        try:
            directory = './Gridworlds/' + argv[0]
            board = pd.read_csv(directory, sep="	", header=None)
            start = (np.where(board == 'S')[0][0], np.where(board == 'S')[1][0])
            board = board\
                .replace('+', Value.COOKIE)\
                .replace("-", Value.GLASS)\
                .replace("X", Value.BARRIER)\
                .replace("S", Value.EMPTY)\
                .values.astype(int)

            gridworld = Gridworld(board, start)
        except Exception as e:
            print(e, command_format)
            continue
        
        try:
            runtime = float(argv[1])  # assuming this is positive
            
            if runtime <= 0:
                raise 
        except:
            print("Exception: Invalid seconds for command argument 2", command_format)
            continue
        
        try:
            reward = float(argv[2])  # assuming this is non-positive
            
            if reward > 0:
                raise 
        except:
            print("Exception: Invalid reward for command argument 3", command_format)
            continue
        
        try:
            transition_model = float(argv[3])  # assuming this is <=1
            
            if transition_model > 1:
                raise 
        except:
            print("Exception: Invalid transition model for command argument 4", command_format)
            continue
        
        try:
            bool = argv[4].lower()
            
            if bool == "true":
                time_based = True
            elif bool == "false":
                time_based = False
            else:
                raise 
        except:
            print("Exception: Invalid boolean for command argument 5", command_format)
            continue
        
        driver = RL(gridworld, runtime, reward, transition_model, time_based)

        driver.start()


def get_input():
    user_input = input("Enter arguments:")
    return user_input.split()


if __name__ == "__main__":
    main(sys.argv[1:])
