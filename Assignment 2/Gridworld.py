from enum import IntEnum
import itertools
class Value(IntEnum):
    EMPTY = 0
    COOKIE = 10
    GLASS = 11
    BARRIER = 12


class Gridworld:
    def __init__(self, gridworld, position):
        self.gridworld = gridworld
        self.position = position

    def _value_str(self, value):
        if value == Value.EMPTY:
            return "0"
        if value == Value.COOKIE:
            return "+"
        if value == Value.GLASS:
            return "-"
        if value == Value.BARRIER:
            return "X"
        return str(value)

    def __str__(self):
        str = ""
        for row, line in enumerate(self.gridworld):
            for col, val in enumerate(line):
                if (row, col) == self.position:
                    str += "S  "
                    continue
                str += "{:<3}".format(self._value_str(val))
            str += "\n"
        return str
