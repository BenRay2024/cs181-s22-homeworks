# Imports.
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import pygame as pg

# uncomment this for animation
#from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_y = None
        self.gravity_updated = False
        self.gravity = None

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE, 2))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_y == None
        self.gravity_updated = False
        self.gravity = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.
        
        epsilon = 0.0001
        alpha = 0.1
        gamma = 0.9

        # If first step
        if self.last_action == None or self.last_state == None or self.last_y == None:
            self.last_action = 0
            self.last_state = self.discretize_state(state)
            self.last_y = state["monkey"]["top"]
        else:
            # Discretize state
            s_new = self.discretize_state(state) # s'
            x_new = s_new[0]
            y_new = s_new[1]

            # s
            x_last = self.last_state[0]
            y_last = self.last_state[1] 

            # Calculate gravity
            y = state["monkey"]["top"]
            if y == self.last_y:
                return self.last_action
            if self.gravity == None:
                self.gravity = y - self.last_y
                self.gravity_updated = True
                print(self.gravity)
                if self.gravity == -1.0:
                    self.gravity = 0
                else:
                    self.gravity = 1
            g = self.gravity

            # Q-Learning update
            Q_next = [self.Q[0,x_new,y_new,g], self.Q[1,x_new,y_new,g]]
            a_new = np.argmax(Q_next) # a'
            self.Q[int(self.last_action),x_last,y_last,g] += alpha * (self.last_reward + gamma * Q_next[a_new] - self.Q[int(self.last_action),x_last,y_last,g])
            
            if npr.rand() < epsilon:
                new_action = int(npr.rand() < 0.5)
            else:
                new_action = a_new

            self.last_action = new_action
            self.last_state = s_new

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)

    # Save history. 
    np.save('hist', np.array(hist))
