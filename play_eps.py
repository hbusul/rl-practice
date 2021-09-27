#!/home/burak/anaconda3/envs/rl/bin/python
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class XOX(gym.Env):
    '''Some description here'''

    def __init__(self):
        self.state: np.ndarray = np.array([0] * 9)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([3] * 9)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state: np.ndarray = np.array([0] * 9)

    def step(self, action):
        err_msg = f'{action} ({action}) invalid'
        assert self.action_space.contains(action), err_msg
        assert self.state[action] == 0, 'square was already full'

        # player makes a move
        self.state[action] = 1
        s = self.state.reshape(3, 3)
        if any([((s[i] == 1).all() or (s[:, i] == 1).all()) for i in range(3)]) or (s.diagonal() == 1).all()\
                or (s[((0, 1, 2), (2, 1, 0))] == 1).all():
            return self.state, 1, True, {}

        # opponent makes the next move
        free_places = np.where(self.state == 0)[0]
        if len(free_places) == 0:
            return self.state, 0.5, True, {}

        opponent_move = free_places[self.np_random.randint(len(free_places))]
        self.state[opponent_move] = 2
        s = self.state.reshape(3, 3)
        if any([((s[i] == 2).all() or (s[:, i] == 2).all()) for i in range(3)]) or (s.diagonal() == 2).all()\
                or (s[((0, 1, 2), (2, 1, 0))] == 2).all():
            reward = -1
            return self.state, -1, True, {}

        if len(free_places) == 1:
            return self.state, 0.5, True, {}
        else:
            return self.state, 0, False, {}

def q(values, state : np.ndarray):
    x = tuple(state.tolist())
    if x not in values:
        values[x] = [np.array([0] * 9), np.array([0] * 9)]
    return values[x]


def main():
    # epsilon-greedy td(0) learning
    dr = 0.99 # discount rate
    lr = 0.01 # learning rate

    number_of_episodes = 10000
    q_table = {}
    xox = XOX()

    sarsa_buf = []
    wins = 0
    loses = 0
    ties = 0

    for i in range(number_of_episodes):
        epsilon = 1 / (i + 1)
        done = False
        while not done:
            cur_state = xox.state.copy()
            q_values, frequencies = q(q_table, cur_state)
            rand = xox.np_random.rand()
            free_places = np.where(xox.state == 0)[0]
            if rand < epsilon:
                act = free_places[xox.np_random.randint(len(free_places))]
            else:
                q_values[xox.state != 0] = -1000
                act = np.argmax(q_values)
            s, r, done, _ = xox.step(act)
            sarsa_buf.append((cur_state, act, r))
            last_two = sarsa_buf[-2:]
            if len(last_two) == 2:
                s1, a1, r1 = sarsa_buf[-2]
                s2, a2, r2 = sarsa_buf[-1]
                # update according to td(0)
                old_q_values, old_frequencies = q(q_table, s1)
                q1 = q(q_table, s1)[0][a1]
                q2 = q(q_table, s2)[0][a2]
                q1 += lr * (r1 + (dr * q2) - q1)
                old_q_values[a1] = q1
                q_table[tuple(s1.tolist())] = [old_q_values, old_frequencies]
            
            #last_three = sarsa_buf[-3:]
            #if len(last_three) == 3:
                #s1, a1, r1 = sarsa_buf[-3]
                #_, _, r2 = sarsa_buf[-2]
                #s3, a3, r3 = sarsa_buf[-1]

                #old_q_values = q(q_table, s1)
                #q1 = q(q_table, s1)[a1]
                #q3 = q(q_table, s3)[a3]
                #
                #q1 += lr * (r1 + (dr * (r2 + (dr * q3))) - q1)
                #old_q_values[a1] = q1
                #q_table[tuple(s1.tolist())] = old_q_values

            if done:
                if r == 1:
                    wins += 1
                elif r == -1:
                    loses += 1
                else:
                    ties += 1
        done = False
        xox.reset()

    print(f'wins: {wins}, loses: {loses}, ties: {ties}')

if __name__ == '__main__':
    main()
