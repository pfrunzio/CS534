import math
import random
import time
from copy import deepcopy
from threading import Timer
from Gridworld import Action, Value


class RL:
    def __init__(self, gridworld, runtime, reward, transition_model, time_based):
        self.gridworld = gridworld
        self.heatmap = deepcopy(gridworld)

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

        self.q_table = dict()

    def start(self):
        print(
            f'Performing RL in {self.runtime} seconds with {self.per_action_reward} reward per action and actions succeeding {self.transition_model * 100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld, '\n')

        epsilon = 1
        decay_rate = 0.999

        random.seed(21)

        # start learning
        return self._rl(epsilon, decay_rate, True)

    def graph_start(self, epsilon, decay_rate, time_decay):
        print(
            f'Performing RL in {self.runtime} seconds with {self.per_action_reward} reward per action and actions succeeding {self.transition_model * 100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld, '\n')
        random.seed(21)
        return self._rl(epsilon, decay_rate, time_decay)

    def get_mean_reward(self, new_time, epsilon, count_episodes):
        self.epsilons.append((new_time, epsilon))
        if len(self.current_rewards) != 0:
            self.mean_rewards.append((new_time, sum(self.current_rewards) / count_episodes))

    def _rl(self, epsilon, decay_rate, time_decay):
        end_time = time.time() + self.runtime
        self.end_time = end_time

        start_time = time.time()

        count_episodes = 0

        # better exploration for part 3/4
        # decay_rate =
        # self.step_size_parameter =

        while time.time() < end_time:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.position

            action = self._select_action(current_state, epsilon)

            # calculate graph data
            # if (time.time() - last_time) >= .025:
            if count_episodes > 100:
                self.get_mean_reward((time.time() - start_time), epsilon, count_episodes)
                self.current_rewards = []
                count_episodes = 0

            while not terminal:
                new_board, reward, terminal = current_gridworld.take_action(action, self.per_action_reward,
                                                                            self.transition_model)

                self.current_rewards.append(reward)

                current_gridworld = new_board
                new_state = current_gridworld.position

                new_action = self._select_action(new_state, epsilon)

                self._update_utility(current_state, action, reward, new_state, new_action)

                self._update_heatmap(current_state)
                current_state = new_state
                action = new_action

            # decay and time-based factors
            if time_decay:
                epsilon *= decay_rate

            # Stops exploring when the time left is less than 1% of the given time and less than 0.1 second
            # TODO: This has a negligible effect on epsilon/exploration, replace with better metric?
            if self.time_based:
                if ((end_time - time.time()) / self.runtime) < 0.1:
                    epsilon = 0

            # update count for graph data
            count_episodes += 1

        # print results
        policy = self._generate_policy()
        print("Policy:")
        print(policy, "\n")

        print("Heatmap:")
        print(self.heatmap, "\n")
        return self.mean_rewards, self.epsilons

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

    def _generate_policy(self):
        policy = self.gridworld

        for row in range(len(self.gridworld.gridworld)):
            for col in range(len(self.gridworld.gridworld[0])):
                state = (row, col)

                if policy[row][col] == Value.EMPTY or policy[row][col] == Value.COOKIE or policy[row][
                    col] == Value.GLASS:

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
        self.heatmap[row][col] += 1
