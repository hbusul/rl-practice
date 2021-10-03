import argparse
import pickle
import sys

import numpy as np

from utils import get_q, update_q, last_action_update_q, multiply_actions
from xox_env import XOX


def test(number_of_episodes):
    '''Play games against agent given number of times and report results'''
    wins = 0
    loses = 0
    ties = 0

    xox = XOX()
    with open('q_table_eps.pckl', 'rb') as f:
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


def train(number_of_episodes, use_model, skip_result, skip_percentage, fixed_epsilon):
    # epsilon-greedy td(0) learning
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
        epsilon = 1 / (i + 1) if fixed_epsilon != 0 else fixed_epsilon
        done = False
        while not done:
            cur_state = xox.state.copy()
            q_values, frequencies = get_q(q_table, cur_state)
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
    with open('q_table_eps.pckl', 'wb') as f:
        pickle.dump(q_table, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'test'])
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--use-model', action='store_true')
    parser.add_argument('--test-after', type=int, default=0)
    parser.add_argument('--skip-printing-training', action='store_true')
    parser.add_argument('--skip-percentage', action='store_true')
    parser.add_argument('--fixed-epsilon', type=float, default=0)
    args = parser.parse_args()
    if args.phase == 'train':
        train(args.episodes, args.use_model, args.skip_printing_training, args.skip_percentage, args.fixed_epsilon)
        if args.test_after:
            test(args.test_after)
    else:
        test(args.episodes)
