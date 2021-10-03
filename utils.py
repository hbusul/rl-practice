import numpy as np


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
