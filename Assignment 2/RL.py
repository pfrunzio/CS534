
class RL():
    def __init__(self, gridworld, runtime, reward, transition_model, time_based):
        self.gridworld = gridworld
        self.runtime = runtime
        self.reward = reward
        self.transition_model = transition_model
        self.time_based = time_based
        self.gamma = 1
        
    def start(self):
        print(f'Performing RL in {self.runtime} seconds with {self.reward} reward per action and actions succeeding {self.transition_model*100} percent of the time {"taking into account" if self.time_based else "ignoring"} remains.\n')
        print("Initial World:")
        print(self.gridworld)
        