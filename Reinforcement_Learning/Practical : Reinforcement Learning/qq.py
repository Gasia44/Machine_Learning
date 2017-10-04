import cv2
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
import numpy as np
import matplotlib.pyplot as plt

class QAgent(Agent):
    def __init__(self):
        super(QAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
        self.episode_reward = np.zeros(500)

        self.grid_look_col = 7
        self.grid_look_row = 7

        self.my_state = 0
        self.alpha = 0.01
        self.gamma = 0.9

        self.epsilon = 0.01 #0.01 0.15
        self.old_episode = 1

        self.policy_dict = {}
        for i in range(10):
            self.policy_dict[i] = np.zeros(((self.grid_look_col * self.grid_look_row), 7))

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0
        self.action_index = 0
        self.current_reward = 0

        self.grid = grid


    def random_argmax(a, array):
        return np.random.choice(np.flatnonzero(array == array.max()))

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        # self.total_reward += self.move(action)

        self.my_state = np.argmax(self.grid[0])
        self.opponent_col = self.my_state
        self.opponent_row = 0
        min_dist = float('inf')

        ##coordinates of the nearest car:
        for col in range(int(self.grid_look_col / 2)):
            for row in range(self.grid_look_row):
                if self.my_state - col > 0:
                    if self.grid[row][self.my_state - col] == 1:
                        if min_dist > row **2 + (self.my_state - col) **2:
                            self.opponent_col = col
                            self.opponent_row = row
                            min_dist = row **2 + (self.my_state - col) **2

                if self.my_state + col < self.grid.shape[1]:
                    if self.grid[row][self.my_state + col] == 1:
                        if min_dist > row ** 2 + (self.my_state - col) ** 2:
                            self.opponent_col = col
                            self.opponent_row = row
                            min_dist = row ** 2 + (self.my_state - col) ** 2

        current_policy = self.policy_dict[self.my_state]


        if np.random.random() < 1 - self.epsilon:
            self.action_index = self.random_argmax(current_policy[self.opponent_row*self.grid_look_row + self.opponent_col][2:6])

        else:
            self.action_index = np.random.choice(4 )

        self.current_reward = self.move(self.getActionsSet()[self.action_index])

        self.total_reward += self.current_reward

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
        self.grid = grid


    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        my_state_new = np.argmax(self.grid[0])
        opponent_col_new = my_state_new
        opponent_row_new = 0
        min_dist = float('inf')

        ##coordinates of the nearest car:
        for col in range(int(self.grid_look_col / 2)):
            for row in range(self.grid_look_row):
                if my_state_new - col > 0:
                    if self.grid[row][my_state_new - col] == 1:
                        if min_dist > row ** 2 + (my_state_new - col) ** 2:
                            opponent_col_new = col
                            opponent_row_new = row
                            min_dist = row ** 2 + (my_state_new - col) ** 2

                if my_state_new + col < self.grid.shape[1]:
                    if self.grid[row][my_state_new + col] == 1:
                        if min_dist > row ** 2 + (my_state_new - col) ** 2:
                            opponent_col_new = col
                            opponent_row_new = row
                            min_dist = row ** 2 + (my_state_new - col) ** 2


        new_policy = self.policy_dict[my_state_new]


        current_policy = self.policy_dict[self.my_state]
        current_policy[self.opponent_row * self.grid_look_row + self.opponent_col][2 + self.action_index] += self.alpha*(self.current_reward + self.gamma* np.max( new_policy[opponent_row_new*self.grid_look_row + opponent_col_new][2:6]) - current_policy[self.opponent_row * self.grid_look_row + self.opponent_col][2 + self.action_index])

        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print("{0}/{1}: {2}".format(episode, iteration, self.total_reward))
        self.episode_reward[episode] = self.total_reward

        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)

if __name__ == "__main__":
    a = QAgent()
    ep = 200
    a.run(True, episodes=ep, draw=True)
    print('-----Episode reward-----\n' + str(a.episode_reward[0:ep+1]))

    print('----------------')
    plt.plot(a.episode_reward[0:ep+1])
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    print('Total reward: ' + str(a.total_reward))
