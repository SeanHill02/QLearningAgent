import numpy as np

from action_value_table import ActionValueTable

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3  # agents actions
GAMMA = 0.95
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON


    def update(self, state, action, reward, next_state, done):
        current_value = self.values.get_value(state, action)
        next_max = max([self.values.get_value(next_state, a) for a in self.actions])
        new_value = current_value + self.step_size * (reward + self.gamma * next_max - current_value)
        self.values.set_value(state, action, new_value)


    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            action = np.random.choice(self.actions)
        else:
            # Exploitation: choose the greedy action
            action = self.get_greedy_action(state)
        return action


    def get_greedy_action(self, state):
        values = [self.values.get_value(state, a) for a in self.actions]
        max_value = max(values)
        best_actions = [a for a, v in enumerate(values) if v == max_value]
        action = np.random.choice(best_actions)
        return action
