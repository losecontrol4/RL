import gymnasium as gym
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.plots import Plots
import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time
from gymnasium.envs.toy_text.utils import categorical_sample
import gymnasium as gym
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from gymnasium.envs.toy_text.blackjack import draw_card, is_bust, is_natural, sum_hand, score, cmp
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt



size = 20 # size 5 is that L shape one   #5, 25, 5O
cost = 0
gamma = .01
slippery = False
max_episode_steps = 2500

np.random.seed(10)
# frozen_lake = gym.make('FrozenLake-v1', render_mode="rgb_array" ,desc=generate_random_map(size=size), is_slippery=slippery, max_episode_steps=max_episode_steps)
# Q, V, pi, Q_track, pi_track = RL(frozen_lake).q_learning(gamma=gamma, min_epsilon=.01, alpha_decay_ratio=.01)


# # # Q-learning
# # print(np.mean(V))
# # Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning()

# # #test policy
# test_scores = TestEnv.test_env(env=frozen_lake, n_iters=10, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))

    
base_env = gym.make('Blackjack-v1', render_mode=None, natural=False)
# Q-learning
blackjack = BlackjackWrapper(base_env)
    
Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning()

#test policy
print(pi)
test_scores = TestEnv.test_env(env=blackjack, n_iters=10000, render=False, pi=pi, user_input=False)
print(np.mean(test_scores))
