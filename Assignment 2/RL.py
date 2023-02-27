import math
import random
import time
from copy import deepcopy

from Gridworld import Action, Value

EXPLORATION_RATE = 0.5


class RL:
    def __init__(self, gridworld, runtime, reward, transition_model, time_based):
        self.gridworld = gridworld
        self.runtime = runtime
        self.per_action_reward = reward
        self.transition_model = transition_model
        self.time_based = time_based
        self.future_reward_discount = 1
        self.step_size_parameter = .9
        self.heatmap = deepcopy(gridworld)

        # (state, action) -> utility
        self.q_table = dict()

    def start(self):
        print(
            f'Performing RL in {self.runtime} seconds with {self.per_action_reward} reward per action and actions succeeding {self.transition_model * 100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld, '\n')

        # start learning
        self._rl()

    def _rl(self):
        end_time = time.time() + self.runtime

        count = 0

        while time.time() < end_time:

            terminal = False
            current_gridworld = self.gridworld
            current_state = current_gridworld.position

            action = self._select_action(current_state)

            while not terminal:
                new_board, reward, terminal = current_gridworld.take_action(action, self.per_action_reward, 1)
                current_gridworld = new_board
                new_state = current_gridworld.position
                
                new_action = self._select_action(new_state)

                self._update_utility(current_state, action, reward, new_state, new_action)

                self._update_heatmap(current_state)
                current_state = new_state
                action = new_action
        
        policy = self._generate_policy()
        print("Policy:")
        print(policy, "\n")
        
        print("Heatmap:")
        print(self.heatmap, "\n")
        return

    def _select_action(self, state):

        return self._epsilon_greedy(EXPLORATION_RATE, state)

    def _update_utility(self, state, action, reward, new_state, new_action):

        # if you want to do SARSA, add the utility from q_table for a random/epsilon greedy action from new_state
        # if you want to do Q-Learning, add the max utility from q_table for any/all actions from new_state

        new_utility = self._sarsa_utility(state, action, reward, new_state, new_action, self.step_size_parameter)

        self.q_table[(state, action)] = new_utility

    def _q_learning_utility(self, state, action, reward, new_state, step_size_parameter):

        current_utility = self._get_utility(state, action)
        new_utility = self._get_utility(new_state, self._get_best_action(new_state))

        return self._calculate_temporal_difference(current_utility, new_utility, reward, step_size_parameter)

    def _sarsa_utility(self, state, action, reward, new_state, new_action, step_size_parameter):

        current_utility = self._get_utility(state, action) #6.264
        new_utility = self._get_utility(new_state, new_action) #-2.736

        return self._calculate_temporal_difference(current_utility, new_utility, reward, step_size_parameter)

    def _calculate_temporal_difference(self, current_utility, new_utility, reward, step_size_parameter):
        return current_utility + \
            (step_size_parameter * (reward + (self.future_reward_discount * new_utility - current_utility))) #-9
#6.264 + .9 * (6.96 + -9)
    #4.428
    def _epsilon_greedy(self, epsilon, state):

        rand = random.random()

        if (rand < epsilon):
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
            # self.q_table[(state, action)] = 0
            return 0

    def _explore(self):
        return random.choice(list(Action))
    
    def _generate_policy(self):
        policy = self.gridworld
        
        for row in range(len(self.gridworld.gridworld)):
            for col in range(len(self.gridworld.gridworld[0])):
                state = (row, col)
                
                if policy[row][col] == 0:
                
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