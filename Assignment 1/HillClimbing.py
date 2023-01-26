from Algorithm import Algorithm

class HillClimbing(Algorithm):

    def __init__(self, board, seconds):
        super().__init__(board)
        self.seconds = seconds

    def start(self):
        print(f'Performing greedy (hill climbing) search for {self.seconds} seconds')
        print("Initial Board:")
        print(self.board)