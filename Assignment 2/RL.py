from Gridworld import Action


class RL:
    def __init__(self, gridworld, runtime, reward, transition_model, time_based):
        self.gridworld = gridworld
        self.runtime = runtime
        self.reward = reward
        self.transition_model = transition_model
        self.time_based = time_based
        self.gamma = 1

        # (state, action) -> utility
        self.q_table = dict()

    def start(self):
        print(f'Performing RL in {self.runtime} seconds with {self.reward} reward per action and actions succeeding {self.transition_model*100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld)

        # start learning
        self._rl()

    def _rl(self):
        # TODO

        # How to use the Gridworld code:
        # new_board, reward, terminal = self.gridworld.take_action(Action.UP, self.reward, self.transition_model)
        # state = self.gridworld.position

        # ^ this should be all you need to call from Gridworld

        return

    def _select_action(self, state):
        # TODO
        return

    def _update_utility(self, state, action, reward, new_state):
        # TODO

        # if you want to do SARSA, add the utility from q_table for a random/epsilon greedy action from new_state
        # if you want to do Q-Learning, add the max utility from q_table for any/all actions from new_state

        return
        