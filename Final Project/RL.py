from Gridworld import Gridworld, Action
import math
import random
import time
import numpy as np
from copy import deepcopy


class RL:
    def __init__(self, gridworld, runtime):
        self.gridworld = gridworld

        self.actions = list(Action)

        if gridworld.hasInventory == False:
            self.actions.pop()
            self.actions.pop()

        # run settings
        self.runtime = runtime
        self.future_reward_discount = 1  # gamma
        self.step_size_parameter = .1  # alpha

        self.q_table = dict()

        self.mean_rewards = [[], [], []]

    def start(self):
        print(
            f'Performing RL in {self.runtime} seconds\n')
        print("Initial World:")
        print(self.gridworld, '\n')

        # random.seed(21)

        # start learning
        return self._rl()

    def _rl(self):
        epsilon = 1

        start_time = time.time()
        end_time = start_time + self.runtime
        self.end_time = end_time

        reward_count = 0
        count_episodes = 0

        while time.time() < end_time:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.get_q_table_state()

            action = self._select_action(current_state, epsilon)

            # calculate graph data
            if count_episodes % 100 == 0 and count_episodes != 0:
                new_time = time.time() - start_time

                time_diff, mean_reward = self._calc_mean_reward_graph()
                self.mean_rewards[0].append(new_time)
                self.mean_rewards[1].append(mean_reward)
                self.mean_rewards[2].append(reward_count/count_episodes)
                # account for time spent calculating mean
                start_time += time_diff
                end_time += time_diff

            while not terminal:
                # change rewards and stuff later
                new_board = current_gridworld.take_action(action)
                reward = 1
                terminal = new_board.is_terminal

                current_gridworld = new_board
                new_state = current_gridworld.get_q_table_state()

                new_action = self._select_action(new_state, epsilon)

                self._update_utility(current_state, action, reward, new_state, new_action)

                current_state = new_state
                action = new_action

                reward_count += reward

            # exploration
            percent_used = 1 - (end_time - time.time()) / self.runtime

            epsilon = 1 - percent_used
            self.step_size_parameter = pow(2, -40 * percent_used) + 0.05
            if percent_used >= 0.8:
                self.step_size_parameter = 0.01
                epsilon = 0

            count_episodes += 1

        # print results
        print(f'Mean reward: {self._calc_mean_reward()}')
        print("done")
        return self.mean_rewards

    def _select_action(self, state, epsilon):
        return self._epsilon_greedy(epsilon, state)

    def _update_utility(self, state, action, reward, new_state, new_action):

        # if you want to do SARSA, add the utility from q_table for a random/epsilon greedy action from new_state
        # if you want to do Q-Learning, add the max utility from q_table for any/all actions from new_state

        new_utility = self._q_learning_utility(state, action, reward, new_state, self.step_size_parameter)
        # new_utility = self._sarsa_utility(state, action, reward, new_state, new_action, self.step_size_parameter)

        self.q_table[(state, action)] = new_utility

    def _q_learning_utility(self, state, action, reward, new_state, step_size_parameter):

        current_utility = self._get_utility(state, action)
        new_utility = self._get_utility(new_state, self._get_best_action(new_state))

        return self._calculate_temporal_difference(current_utility, new_utility, reward, step_size_parameter)

    def _sarsa_utility(self, state, action, reward, new_state, new_action, step_size_parameter):

        current_utility = self._get_utility(state, action)
        new_utility = self._get_utility(new_state, new_action)

        return self._calculate_temporal_difference(current_utility, new_utility, reward, step_size_parameter)

    def _calculate_temporal_difference(self, current_utility, new_utility, reward, step_size_parameter):
        return current_utility + \
            (step_size_parameter * (reward + (self.future_reward_discount * new_utility - current_utility)))  # -9

    def _epsilon_greedy(self, epsilon, state):

        rand = random.random()

        if rand < epsilon:
            return self._explore()
        else:
            return self._get_best_action(state)

    def _get_best_action(self, state):

        best_utility = -math.inf
        best_action = Action.UP

        for action in self.actions:

            current_utility = self._get_utility(state, action)

            if current_utility > best_utility:
                best_utility = current_utility
                best_action = action

        return best_action

    def _get_utility(self, state, action):
        try:
            return self.q_table[(state, action)]
        except KeyError as e:
            return 0

    def _explore(self):
        return random.choice(self.actions)

    def _calc_mean_reward(self):
        trial_count = 0
        total_reward = 0

        # average over 100 runs
        while trial_count < 100:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.get_q_table_state()

            trial_reward = 0

            while not terminal:
                action = self._get_best_action(current_state)

                new_board = current_gridworld.take_action(action)
                reward = 1
                terminal = new_board.is_terminal

                trial_reward += reward
                current_gridworld = new_board
                current_state = current_gridworld.get_q_table_state()

            total_reward += trial_reward
            trial_count += 1

        return total_reward / trial_count

    def _calc_mean_reward_graph(self):
        start_time = time.time()
        trial_count = 0
        total_reward = 0

        # average over 100 runs
        while trial_count < 20:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.get_q_table_state()

            trial_reward = 0

            while not terminal:
                action = self._get_best_action(current_state)

                new_board = current_gridworld.take_action(action)
                reward = 1
                terminal = new_board.is_terminal

                trial_reward += reward
                current_gridworld = new_board
                current_state = current_gridworld.get_q_table_state()

            total_reward += trial_reward
            trial_count += 1

        return time.time() - start_time, total_reward / trial_count
