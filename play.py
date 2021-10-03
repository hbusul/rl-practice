'''
Provides TD(0) UCB update method for XOX environment.
You can run this file directly such as:
```
> python3 play.py train --episodes 10000 --after-test 1000 --use-model
```
Percentage is written to stderr by default, however you can skip printing
it with `--skip-percentage`.


or you can import train method and use this file as library
```
from play import test, train
episodes = 10000
use_model = True
skip_result = False
skip_percentage = False
train(episodes, use_model, skip_result, skip_percentage)
test(1000)
```
'''


import argparse
import pickle
import sys

import numpy as np

from xox_env import XOX


def get_q(values, state: np.ndarray):
    '''Given a q table and a state returns state-action value
    if found, else creates and returns an inital value'''
    key = tuple(state.tolist())
    if key not in values:
        values[key] = [np.array([0.2] * 9), np.array([0] * 9)]
    return values[key]


def update_q(q_table, s1, a1, r1, s2, a2, lr, dr):
    '''TD(0) Update'''
    old_q_values, old_frequencies = get_q(q_table, s1)
    q1 = get_q(q_table, s1)[0][a1]
    q2 = get_q(q_table, s2)[0][a2]
    q1 += lr * (r1 + (dr * q2) - q1)
    old_q_values[a1] = q1
    old_frequencies[a1] += 1
    q_table[tuple(s1.tolist())] = [old_q_values, old_frequencies]


def last_action_update_q(q_table, s1, a1, r1, lr):
    '''TD(0) Update for the last action'''
    old_q_values, old_frequencies = get_q(q_table, s1)
    old_frequencies[a1] += 1
    q1 = old_q_values[a1]
    q1 += lr * (r1 - q1)
    old_q_values[a1] = q1
    q_table[tuple(s1.tolist())] = [old_q_values, old_frequencies]


def multiply_actions(s1, a1, r1, s2, a2, r2):
    '''When you rotate or mirror XOX game, that is also a valid game and
    action values of rotation and mirroring should also be the same.
    Using this fact, we can create multiple state, action, reward pairs
    from a single state, action, reward and use those values in training.'''
    remap_indices = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        np.array([6, 7, 8, 3, 4, 5, 0, 1, 2]),
        np.array([2, 1, 0, 5, 4, 3, 8, 7, 6]),
        np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
    ]

    result = {}
    for remap in remap_indices:
        sx, ax = s1[remap], remap[a1]
        sy, ay = s2[remap], remap[a2]
        key = tuple([*sx.tolist(), ax, *sy.tolist(), ay])
        if key not in result:
            result[key] = (sx, ax, r1, sy, ay, r2)
    return result.values()


def train(number_of_episodes, use_model, skip_result, skip_percentage):
    '''Train agent with given parameters'''
    # td(0) updates with Upper Confidence Bound
    dr = 0.99  # discount rate
    lr = 0.01  # learning rate

    q_table = {}
    xox = XOX()

    sarsa_buf = []
    wins = 0
    loses = 0
    ties = 0

    ten_percent = number_of_episodes // 10

    for i in range(number_of_episodes):
        if not skip_percentage and ((i + 1) % ten_percent == 0 or i == number_of_episodes - 1):
            print('%.1f%%' % (((i + 1) / number_of_episodes) * 100), file=sys.stderr)
        done = False
        while not done:
            cur_state = xox.state.copy()
            q_values, frequencies = get_q(q_table, cur_state)
            u_values = np.sqrt(
                np.log(i + 2) / (np.array(frequencies, dtype=np.float32) + 1e-4)
            ) * np.sqrt(2)
            q_values[xox.state != 0] = -1000
            act = np.argmax(q_values + u_values)
            _, r, done, _ = xox.step(act)
            sarsa_buf.append((cur_state, act, r))
            last_two = sarsa_buf[-2:]
            if len(last_two) == 2:
                sar1 = sarsa_buf[-2]
                sar2 = sarsa_buf[-1]

                tuples = multiply_actions(*sar1, *sar2) if use_model else [(*sar1, *sar2)]
                for sarsa in tuples:
                    update_q(q_table, sarsa[0], sarsa[1], sarsa[2], sarsa[3], sarsa[4], lr, dr)

            if done:
                sar1 = sarsa_buf[-1]
                tuples = multiply_actions(*sar1, *sar1) if use_model else [(*sar1, *sar1)]
                for sarsa in tuples:
                    last_action_update_q(q_table, sarsa[0], sarsa[1], sarsa[2], lr)

                if r == 1:
                    wins += 1
                elif r == -1:
                    loses += 1
                else:
                    ties += 1
        done = False
        xox.reset()

    if not skip_result:
        print(f'wins: {wins}, loses: {loses}, ties: {ties}')
    with open('q_table.pckl', 'wb') as f:
        pickle.dump(q_table, f)


def test(number_of_episodes):
    '''Play games against agent given number of times and report results'''
    wins = 0
    loses = 0
    ties = 0

    xox = XOX()
    with open('q_table.pckl', 'rb') as f:
        q_table = pickle.load(f)

    for _ in range(number_of_episodes):
        done = False
        while not done:
            cur_state = xox.state.copy()
            q_values, _ = get_q(q_table, cur_state)
            q_values[cur_state != 0] = -100
            act = np.argmax(q_values)
            _, reward, done, _ = xox.step(act)

            if done:
                if reward == 1:
                    wins += 1
                elif reward == -1:
                    loses += 1
                else:
                    ties += 1
        xox.reset()

    print(f'wins: {wins}, loses: {loses}, ties: {ties}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'test'])
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--use-model', action='store_true')
    parser.add_argument('--test-after', type=int, default=0)
    parser.add_argument('--skip-printing-training', action='store_true')
    parser.add_argument('--skip-percentage', action='store_true')
    args = parser.parse_args()
    if args.phase == 'train':
        train(args.episodes, args.use_model, args.skip_printing_training, args.skip_percentage)
        if args.test_after:
            test(args.test_after)
    else:
        test(args.episodes)
