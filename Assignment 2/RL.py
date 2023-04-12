import math
import random
import time
import numpy as np
from copy import deepcopy
from threading import Timer
from Gridworld import Action, Value
from itertools import chain, combinations


class RL:
    def __init__(self, gridworld, runtime, reward, transition_model, time_based):
        self.gridworld = gridworld
        self.heatmaps = dict()

        # run settings
        self.runtime = runtime
        self.per_action_reward = reward
        self.transition_model = transition_model
        self.time_based = time_based
        self.future_reward_discount = 1  # gamma
        self.step_size_parameter = .1  # alpha

        # benchmarking data
        self.mean_rewards = []
        self.current_rewards = []
        self.epsilons = []
        self.graph = False

        self.q_table = dict()

    def start(self):
        print(
            f'Performing RL in {self.runtime} seconds with {self.per_action_reward} reward per action and actions succeeding {self.transition_model * 100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld, '\n')

        epsilon = 1
        decay_rate = 0.999

        random.seed(21)

        self.graph = False

        # start learning
        return self._rl(epsilon, decay_rate, True)

    def graph_start(self, epsilon, decay_rate, better_exploration):
        print(
            f'Performing RL in {self.runtime} seconds with {self.per_action_reward} reward per action and actions succeeding {self.transition_model * 100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld, '\n')
        self.graph = True
        random.seed(21)
        return self._rl(epsilon, decay_rate, better_exploration)

    def _rl(self, epsilon, decay_rate, better_exploration):
        start_time = time.time()
        last_time = start_time
        end_time = start_time + self.runtime
        self.end_time = end_time

        count_episodes = 0
        total_episodes = 0
        all_changes = set()

        linear_area = math.sqrt(len(self.gridworld) * len(self.gridworld[0]))

        while time.time() < end_time:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.get_state()

            action = self._select_action(current_state, epsilon)

            # calculate correct graph data
            if (time.time() - last_time) >= .05:
                self.get_mean_reward((time.time() - start_time), epsilon, count_episodes)
                self.current_rewards = []
                count_episodes = 0
                last_time = time.time()

            # calculate graph data
            # if self.graph and count_episodes % 100 == 0:
            #     time_diff = self._calc_mean_reward(time.time() - start_time, epsilon)
            #
            #     # account for time spent calculating mean
            #     start_time += time_diff
            #     end_time += time_diff

            while not terminal:
                new_board, reward, terminal = current_gridworld.take_action(action, self.per_action_reward,
                                                                            self.transition_model)

                self.current_rewards.append(reward)

                current_gridworld = new_board
                new_state = current_gridworld.get_state()

                new_action = self._select_action(new_state, epsilon)

                self._update_utility(current_state, action, reward, new_state, new_action)

                self._update_heatmap(current_state)
                current_state = new_state
                action = new_action

            epsilon *= decay_rate

            # better exploration for part 3
            if better_exploration:
                # epsilon = max(1 - total_episodes / (50 * linear_area), 0.01)
                epsilon = pow(2, (-3.32 / (500 * linear_area)) * total_episodes)

            # better exploration for part 4
            if self.time_based:
                percent_used = 1 - (end_time - time.time()) / self.runtime

                epsilon = 1 - 3 * percent_used
                self.step_size_parameter = pow(2, -40 * percent_used) + 0.05
                if percent_used >= 0.2:
                    self.step_size_parameter = 0.01
                    epsilon = 0

            # update count for graph data
            count_episodes += 1
            total_episodes += 1

            # add changes to create policies later
            for change in current_gridworld.changes:
                all_changes.add(change)

        # print results
        self._print_policies(all_changes)

        # print("Heatmap:")
        # print(self.heatmap, "\n")

        print(f'Mean Reward {np.mean([x[1] for x in self.mean_rewards])}\n')

        return self.mean_rewards, self.epsilons

    def _print_policies(self, changes):
        for subset in self._powerset(changes):
            change = sorted(subset)
            policy = self._generate_policy(tuple(change))

            readable_changes = list(map(self._render_point, change))

            print(f'Policy for State: {readable_changes}')
            print(policy)

            print("Heatmap:")
            print(self.heatmaps[tuple(change)], "\n")

    def _render_point(self, position):
        row = position[0]
        col = position[1]
        return row, col, self.gridworld._value_str(self.gridworld[row][col])

    def _powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def get_mean_reward(self, new_time, epsilon, count_episodes):
        self.epsilons.append((new_time, epsilon))
        if len(self.current_rewards) != 0:
            self.mean_rewards.append((new_time, sum(self.current_rewards) / count_episodes))

    def _calc_mean_reward(self, new_time, epsilon):
        start_time = time.time()
        trial_count = 0
        total_reward = 0

        # average over 100 runs
        while trial_count < 20:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.get_state()

            trial_reward = 0

            while not terminal:
                action = self._get_best_action(current_state)
                new_board, reward, terminal = current_gridworld.take_action(action, self.per_action_reward,
                                                                            self.transition_model)

                trial_reward += reward
                current_gridworld = new_board
                current_state = current_gridworld.get_state()
                if trial_reward < -1:
                    break

            total_reward += trial_reward
            trial_count += 1

        # plot points
        self.mean_rewards.append((new_time, total_reward / trial_count))
        self.epsilons.append((new_time, epsilon))

        # return the amount of time the calculation took
        return time.time() - start_time

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

        for action in Action:

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
        return random.choice(list(Action))

    def _generate_policy(self, changes):
        policy = deepcopy(self.gridworld)
        for change in changes:
            policy = policy.move_to(change, 0)[0]

        for row in range(len(policy.gridworld)):
            for col in range(len(policy.gridworld[0])):
                state = (row, col, changes)

                if policy[row][col] == Value.EMPTY:

                    best_action = self._get_best_action(state)

                    if best_action == Action.UP:
                        policy[row][col] = Value.UP_ARROW
                    elif best_action == Action.RIGHT:
                        policy[row][col] = Value.RIGHT_ARROW
                    elif best_action == Action.DOWN:
                        policy[row][col] = Value.DOWN_ARROW
                    elif best_action == Action.LEFT:
                        policy[row][col] = Value.LEFT_ARROW

        return policy

    def _update_heatmap(self, state):
        row = state[0]
        col = state[1]
        changes = state[2]

        # set heatmap if one does not exist
        if changes not in self.heatmaps:
            new_heatmap = deepcopy(self.gridworld)
            for change in changes:
                new_heatmap = new_heatmap.move_to(change, 0)[0]
            self.heatmaps[changes] = new_heatmap

        self.heatmaps[changes][row][col] += 1
