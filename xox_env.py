'''
Provides XOX class which is a subclass of gym.Env.
Start using it via
```
x = XOX()
state, reward, done = x.step(2)
```
Locations are:

|0|1|2|
|3|4|5|
|6|7|8|

Opponent step is taken by uniform random, let's assume
it took action 0, board becomes:

|o| |x|
| | | |
| | | |

The state is the flattened version of the board.

|o| |x| | | | | | | | |

Numerically,

|2|0|1|0|0|0|0|0|0|0|0|

Where, 1 is your actions and 2 is your opponent's actions.
'''

import gym
from gym import spaces
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
        state = self.state.reshape(3, 3)
        mask = (state == 1)
        out = mask.all(0).any() or mask.all(1).any() or np.diag(mask).all()
        out = out or mask[((0, 1, 2), (2, 1, 0))].all()
        if out:
            return self.state, 1, True, {}

        # opponent makes the next move
        free_places = np.where(self.state == 0)[0]
        if len(free_places) == 0:
            return self.state, 0.5, True, {}

        opponent_move = free_places[self.np_random.randint(len(free_places))]
        self.state[opponent_move] = 2
        state = self.state.reshape(3, 3)
        mask = (state == 2)
        out = mask.all(0).any() or mask.all(1).any() or np.diag(mask).all()
        out = out or mask[((0, 1, 2), (2, 1, 0))].all()
        if out:
            return self.state, -1, True, {}

        if len(free_places) == 1:
            return self.state, 0.5, True, {}

        return self.state, 0, False, {}
