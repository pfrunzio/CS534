from functools import cache

class Board:

    def __init__(self, board):
        self.board = board

    @cache
    def heuristic(self, sliding, weighted):
        return 0;

    def neighbors(self):
        return self