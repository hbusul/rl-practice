import argparse
import pickle

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


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
        mask = (s == 1)
        out = mask.all(0).any() or mask.all(1).any() or np.diag(mask).all() or mask[((0, 1, 2), (2, 1, 0))].all()
        if out:
            return self.state, 1, True, {}

        # opponent makes the next move
        free_places = np.where(self.state == 0)[0]
        if len(free_places) == 0:
            return self.state, 0.5, True, {}

        opponent_move = free_places[self.np_random.randint(len(free_places))]
        self.state[opponent_move] = 2
        s = self.state.reshape(3, 3)
        mask = (s == 2)
        out = mask.all(0).any() or mask.all(1).any() or np.diag(mask).all() or mask[((0, 1, 2), (2, 1, 0))].all()
        if out:
            reward = -1
            return self.state, -1, True, {}

        if len(free_places) == 1:
            return self.state, 0.5, True, {}
        else:
            return self.state, 0, False, {}


def q(values, state: np.ndarray):
    x = tuple(state.tolist())
    if x not in values:
        values[x] = [np.array([0.2] * 9), np.array([0] * 9)]
    return values[x]


def train():
    # td(0) updates with Upper Confidence Bound
    dr = 0.99  # discount rate
    lr = 0.01  # learning rate

    number_of_episodes = 1000000
    q_table = {}
    xox = XOX()

    sarsa_buf = []
    wins = 0
    loses = 0
    ties = 0

    for i in range(number_of_episodes):
        if i % 10000 == 0:
            print(i / number_of_episodes)
        done = False
        while not done:
            cur_state = xox.state.copy()
            q_values, frequencies = q(q_table, cur_state)
            #print('----\n', q_values, '\n', frequencies)
            u_values = np.sqrt(np.log(i + 2) / (np.array(frequencies, dtype=np.float32) + 1e-4)) * np.sqrt(2)
            #print(u_values, '\n----')
            rand = xox.np_random.rand()
            free_places = np.where(xox.state == 0)[0]
            q_values[xox.state != 0] = -1000
            # print()
            act = np.argmax(q_values + u_values)
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
                old_frequencies[a1] += 1
                q_table[tuple(s1.tolist())] = [old_q_values, old_frequencies]

            if done:
                s1, a1, r1 = sarsa_buf[-1]
                old_q_values, old_frequencies = q(q_table, s1)
                old_frequencies[a1] += 1
                q1 = old_q_values[a1]
                q1 += lr * (r1 - q1)
                old_q_values[a1] = q1
                q_table[tuple(s1.tolist())] = [old_q_values, old_frequencies]

                if r == 1:
                    wins += 1
                elif r == -1:
                    loses += 1
                else:
                    ties += 1
        done = False
        xox.reset()

    print(f'q table len: {len(q_table)}')
    print(f'wins: {wins}, loses: {loses}, ties: {ties}')
    with open('q_table.pckl', 'wb') as f:
        pickle.dump(q_table, f)


def test():
    number_of_episodes = 10000

    wins = 0
    loses = 0
    ties = 0

    xox = XOX()
    with open('q_table.pckl', 'rb') as f:
        q_table = pickle.load(f)

    print(q(q_table, xox.state))
    for _ in range(number_of_episodes):
        done = False
        while not done:
            cur_state = xox.state.copy()
            q_values, frequencies = q(q_table, cur_state)
            act = np.argmax(q_values)
            s, r, done, _ = xox.step(act)

            if done:
                if r == 1:
                    wins += 1
                elif r == -1:
                    loses += 1
                else:
                    ties += 1
        xox.reset()

    print(f'wins: {wins}, loses: {loses}, ties: {ties}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'test'])
    args = parser.parse_args()
    if args.phase == 'train':
        train()
    else:
        test()
