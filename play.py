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

from collections import defaultdict

import numpy as np

from utils import get_q, update_q, last_action_update_q, multiply_actions
from xox_env import XOX


def pick_action_with_ucb(q_table, state, time):
    q_values, frequencies = get_q(q_table, state)
    u_values = np.sqrt(
        np.log(time + 2) / (np.array(frequencies, dtype=np.float32) + 1e-4)
    ) * np.sqrt(2)
    q_values[state != 0] = -1000
    act = np.argmax(q_values + u_values)
    return act


def train(number_of_episodes, use_model, skip_result, skip_percentage):
    '''Train agent with given parameters'''
    # td(0) updates with Upper Confidence Bound
    discount = 0.99
    learning_rate = 0.01

    q_table = {}
    xox = XOX()

    sarsa_buf = []
    stats = defaultdict(lambda : 0)

    ten_percent = number_of_episodes // 10

    for i in range(number_of_episodes):
        if not skip_percentage and ((i + 1) % ten_percent == 0 or i == number_of_episodes - 1):
            print('%.1f%%' % (((i + 1) / number_of_episodes) * 100), file=sys.stderr)
        done = False
        while not done:
            cur_state = xox.state.copy()
            act = pick_action_with_ucb(q_table, xox.state, i)
            _, reward, done, _ = xox.step(act)
            sarsa_buf.append((cur_state, act, reward))
            last_two = sarsa_buf[-2:]
            if len(last_two) == 2:
                sar1 = sarsa_buf[-2]
                sar2 = sarsa_buf[-1]

                tuples = multiply_actions(*sar1, *sar2) if use_model else [(*sar1, *sar2)]
                for sarsa in tuples:
                    update_q(q_table, *sarsa[:5], learning_rate, discount)

            if done:
                sar1 = sarsa_buf[-1]
                tuples = multiply_actions(*sar1, *sar1) if use_model else [(*sar1, *sar1)]
                for sarsa in tuples:
                    last_action_update_q(q_table, sarsa[0], sarsa[1], sarsa[2], learning_rate)

                stats[reward] += 1
        done = False
        xox.reset()

    if not skip_result:
        print(f'wins: {stats[1]}, loses: {stats[-1]}, ties: {stats[0.5]}')
    with open('q_table.pckl', 'wb') as q_table_file:
        pickle.dump(q_table, q_table_file)


def test(number_of_episodes):
    '''Play games against agent given number of times and report results'''
    wins = 0
    loses = 0
    ties = 0

    xox = XOX()
    with open('q_table.pckl', 'rb') as q_table_file:
        q_table = pickle.load(q_table_file)

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
