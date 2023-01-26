from abc import abstractmethod, ABC


class Algorithm(ABC):

    def __init__(self, board):
        self.board = board

    # Driver method for algorithms
    @abstractmethod
    def start(self):
        pass
