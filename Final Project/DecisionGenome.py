from Gridworld import Action, TileValue, Inventory
from astar_search import a_star

number_of_conditions = 6


class DecisionGenome:

    def __init__(self, list_of_conditions, list_of_values, list_of_actions, default_action, eat_threshold):

        self.actions = list(Action)
        self.movement_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.list_of_conditions = list_of_conditions
        self.list_of_values = list_of_values
        self.list_of_actions = list_of_actions
        self.default_action = default_action
        self.eat_threshold = eat_threshold
        self.mutation_rate = 0.3
        self.num_parents = 15
        self.num_of_tables = 2
        self.number_of_conditions = 6
        self.number_of_actions = 5  # include default

    def generate_next_moves(self, gridworld):
        #print(f"default action : {self.default_action}")
        increment_number = 0
        while (increment_number < len(self.list_of_conditions)):
            action = self.get_condtion(gridworld, increment_number)
            increment_number = increment_number +1
            if action and action is not None:
                #print(f"action : {action}")
                return action
        return [self.default_action]

    def get_condtion(self, gridworld, increment_number):
        condition_number = self.list_of_conditions[increment_number]
        value_number = self.list_of_values[increment_number]
        action_number = self.list_of_actions[increment_number]
        if condition_number == 0:
            if gridworld.health > value_number:
                return self.get_action(action_number,gridworld)
            else:
                return None
        elif condition_number == 1:
            if gridworld.health <= value_number:
                return self.get_action(action_number,gridworld)
            else:
                return None
        elif condition_number == 2:
            if gridworld.hydration > value_number:
                return self.get_action(action_number,gridworld)
            else:
                return None
        elif condition_number == 3:
            if gridworld.hydration < value_number:
                return self.get_action(action_number,gridworld)
            else:
                return None
        elif condition_number == 4:
            if gridworld.gridworld[gridworld.pos[0]][gridworld.pos[1]] == TileValue.FOOD:
                return self.get_action(action_number,gridworld)
            else:
                return None
        elif condition_number == 5:
            if gridworld.gridworld[gridworld.pos[0]][gridworld.pos[1]] == TileValue.WATER:
                return self.get_action(action_number,gridworld)
            else:
                return None
        else:
            return [self.default_action]

    def get_action(self, action_number, gridworld):
        if action_number == 0:
            return self.get_direction_to_nearest_food(gridworld)
        elif action_number == 1:
            return self.get_direction_to_nearest_water(gridworld)
        elif action_number == 2:
            return self.get_direction_to_nearest_boar(gridworld)
        elif action_number == 3:
            return self.get_direction_to_nearest_killed_boar(gridworld)
        elif action_number == 4:
            if gridworld.inventory is not Inventory.EMPTY:
                return [Action.USE_INVENTORY]
            else:
                return None
        else:
            return [self.default_action]

    def get_direction_to_nearest_food(self, gridworld):
        path = self.search_for(gridworld, TileValue.FOOD)
        if path:
            if(gridworld.health < self.eat_threshold):
                path.append(Action.USE_TILE)
            else:
                path.append(Action.PICK_UP_ITEM)
        return path

    def get_direction_to_nearest_water(self, gridworld):
        return self.search_for(gridworld, TileValue.WATER)

    def get_direction_to_nearest_boar(self, gridworld):
        path = self.search_for(gridworld, TileValue.BOAR)
        if path:
            path.append(path[len(path)-1])
            if(gridworld.health < self.eat_threshold):
                path.append(Action.USE_TILE)
            else:
                path.append(Action.PICK_UP_ITEM)
        return self.search_for(gridworld, TileValue.BOAR)

    def get_direction_to_nearest_killed_boar(self, gridworld):
        return self.search_for(gridworld, TileValue.KILLED_BOAR)

    def search_for(self, gridworld, item):
        # Check if the item is in the world
        if any(item in x for x in gridworld):
            # check if we are on top of the item we are looking for
            if gridworld.gridworld[gridworld.pos[0]][gridworld.pos[1]] == item:
                if gridworld.health < self.eat_threshold:
                    return [Action.USE_TILE]
                else:
                    return [Action.PICK_UP_ITEM]
            else:
                # search for closest item
                closest_item = (99999999999999999999999999999999, 9999999999999999999999999)
                closest_item_distance = 99999999999999999999999
                closest_item_direction = Action.UP
                closest_item_path = []
                for row in range(len(gridworld.gridworld.gridworld)):
                    for column in range(len(gridworld.gridworld.gridworld[row])):
                        if gridworld[row][column] == item:
                            item_path = a_star(gridworld.gridworld.gridworld, gridworld.pos, closest_item)
                            if item_path is not None and len(item_path) < closest_item_distance:
                                closest_item = (row, column)
                                closest_item_direction = item_path[0]
                                closest_item_path = item_path
                                closest_item_distance = len(item_path)
                return closest_item_path
        # If nothing, invalid search return None go to next condition
        else:
            return None
