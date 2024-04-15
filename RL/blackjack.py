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


np.random.seed(10)

 
# V, V_track, pi = Planner(blackjack.P).value_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
def custom_grid_search(paramater, learning_func, values, output_type):
    
    np.random.seed(10)
  
    output = []
    
    base_env = gym.make('Blackjack-v1', render_mode=None, natural=False)
    blackjack = BlackjackWrapper(base_env)
    
    for val in values:
    
        
        old = time.time()
        
        if paramater == "gamma":
            if learning_func == "v":
                plan = Planner(blackjack.P)
                V, V_track, pi = plan.value_iteration(gamma=val, n_iters=n_iters, theta=theta)
            elif learning_func == "p":
                plan = Planner(blackjack.P)
                V, V_track, pi = plan.policy_iteration(gamma=val, n_iters=n_iters, theta=theta)
                
        elif paramater == "n_iters":
            if learning_func == "v":
                plan = Planner(blackjack.P)
                V, V_track, pi = plan.value_iteration(gamma=gamma, n_iters=val, theta=theta)
            elif learning_func == "p":
                plan = Planner(blackjack.P)
                V, V_track, pi = plan.policy_iteration(gamma=gamma, n_iters=val, theta=theta)
                
        elif paramater == "theta":
            if learning_func == "v":
                plan = Planner(blackjack.P)
                V, V_track, pi = plan.value_iteration(gamma=gamma, n_iters=n_iters, theta=val)
            elif learning_func == "p":
                plan = Planner(blackjack.P)
                V, V_track, pi = plan.policy_iteration(gamma=gamma, n_iters=n_iters, theta=val)     
                
        new = time.time()
        wall_time = new - old

                
        if(output_type == "mean test score"):
            test_score = TestEnv.test_env(blackjack, desc=None, render=False, n_iters=100000, pi=pi, user_input=False)
            test_score = round(np.mean(test_score), 2)
            output.append(test_score)
        elif(output_type == "V mean"):
            output.append(np.mean(V))
        elif(output_type) == "time":
            output.append(wall_time)
    return output
 
 
nums = np.arange(2, 5, 1)

gamma = .99
n_iters = 20
theta = 1e-10
# nums = [.01, .3, .5, .7, .9, .999]

v = custom_grid_search("n_iters", "v", nums, "V mean")
p = custom_grid_search("n_iters", "p", nums, "V mean")


plt.plot(nums,v,
                color="r", label = "vi")
plt.plot(nums,p,
                color="b", label = "pi")

plt.title("Iterations v Mean Test Score")
plt.xlabel("Iterations")
plt.ylabel("Mean Test Score")
plt.grid()
plt.tight_layout()
plt.legend(loc='best')
# plt.savefig("graphs/bj/iter_win_rate_large_gamma")
plt.show()
plt.close()


# gammas = [.01, .99]
# for i, g in enumerate(gammas):
#     gamma = g
 
 
#     v = custom_grid_search("n_iters", "v", nums, "mean test score")
#     p = custom_grid_search("n_iters", "p", nums, "mean test score")
#     plt.plot(nums,v,
#                     color="b", label = "vi small")
#     plt.plot(nums,p,
#                     color="aqua", label = "pi small")

#     size = 25
#     cost = -1/(25*25) 
#     v = custom_grid_search("n_iters", "v", nums, "mean test score")
#     p = custom_grid_search("n_iters", "p", nums, "mean test score")
#     plt.plot(nums,v,
#                     color="green", label = "vi medium")
#     plt.plot(nums,p,
#                     color="lime", label = "pi medium")

#     size = 50
#     cost = -1/(50*50) 
#     v = custom_grid_search("n_iters", "v", nums, "mean test score")
#     p = custom_grid_search("n_iters", "p", nums, "mean test score")
#     plt.plot(nums,v,
#                     color="r", label = "vi large")
#     plt.plot(nums,p,
#                     color="coral", label = "pi large")

#     plt.title("Iterations v Mean Test Score: Not slippery and Gamma: " + str(gamma))
#     plt.xlabel("Iterations")
#     plt.ylabel("Mean Test Score")
#     plt.grid()
#     plt.tight_layout()
#     plt.legend(loc='best')
#     if i == 0:
#         named_size = "small_gamma"
#     else: 
#         named_size = "large_gamma"
#     plt.savefig("graphs/frozen_lake/" + named_size +"_no_slip_with_cost")
#     plt.close()



# run VI


# print(np.mean(V))
# #test policy
# test_scores = TestEnv.test_env(env=blackjack, n_iters=1000, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))

# # Q-learning
# print(np.mean(V))
# Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning()

# #test policy
# test_scores = TestEnv.test_env(env=blackjack, n_iters=100, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))

