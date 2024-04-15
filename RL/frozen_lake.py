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


#add params to Q learning
#PI takes longer (almost 3 times on average for large frozen lake (size 30))
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, cost, reward=1):
        super().__init__(env)
        self.reward_range = range(-100,100)

        
      
        for s in self.P:
            for a in self.P[s]:
                
                for i, outcome in enumerate(self.P[s][a]):
                    if self.P[s][a][i][2] == 0:
                        temp = list(self.P[s][a][i])
                        temp[2] = cost
                        self.P[s][a][i] = tuple(temp)
                    else:
                        temp = list(self.P[s][a][i])
                        temp[2] = reward
                        self.P[s][a][i] = tuple(temp)
                  
                  
np.random.seed(10)
size = 5 # size 5 is that L shape one   #5, 25, 5O
cost = 0
reward = 1
gamma = .99
n_iters = 2000
theta = 1e-150
slippery = False
max_episode_steps = 2500



def run_frozen_lake(learning_func, dir, show = False):
    np.random.seed(10)
    frozen_lake = gym.make('FrozenLake-v1', render_mode="rgb_array" ,desc=generate_random_map(size=size), is_slippery=slippery, max_episode_steps=max_episode_steps) # gym.make('FrozenLake8x8-v1', render_mode="rgb_array", is_slippery=False)

    mask = []

    for line in frozen_lake.desc:
        mask_line = []
        for place in line:
            mask_line.append(place.decode('UTF-8') == "H" or place.decode('UTF-8') == "G")
        mask.append(mask_line)



    CustomRewardWrapper(frozen_lake, cost, reward)

   

    if learning_func == "v":
        func_name = "Value iteration"
        plan = Planner(frozen_lake.P)
        V, V_track, pi = plan.value_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
    elif learning_func == "p":
        func_name = "Policy iteration"
        plan = Planner(frozen_lake.P)
        V, V_track, pi = plan.policy_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
    elif learning_func == "q":
        func_name = "Q learning"
        Q, V, pi, Q_track, pi_track = RL(frozen_lake).q_learning()

 
    test_score = TestEnv.test_env(frozen_lake, desc=None, render=False, n_iters=1000, pi=pi, user_input=False)
    test_score = round(np.mean(test_score), 2)


    # map_size=(size, size)
    def adjust_render(render, size):
        pixels = render.shape[0] // size * size
        return render[:pixels, :pixels, :]


    def plot_main(env, V, pi, cost, mask):
        pi = [pi[key] for key in range(env.observation_space.n)]
        # Plot the policy
        reshaped_V = V.reshape(size, size)
        directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        hmax = sns.heatmap(
            reshaped_V,
            annot=np.reshape([directions[val] for val in pi], (size, size)),
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
            mask=mask,   #reshaped_V < np.finfo(float).eps, # Minimum float number on the machine
            zorder = 2,
        )
        env.reset()
        hmax.set(title= func_name + "\n" + "Size: " + str(size) + " | Cost: " + str(cost) +  " | Reward: " + str(reward) + " | Mean Test Score: " + str(test_score) + " | Max Steps: " + str(max_episode_steps) + "\nGamma: " + str(gamma) + " | Theta: " + str(theta) + " | Slippery: " + str(slippery) + " | Max Iterations: " + str(n_iters))
        hmax.imshow(adjust_render(env.render(), size),
                    aspect = hmax.get_aspect(),
                    extent = hmax.get_xlim() + hmax.get_ylim(),
                    zorder = 1)
        plt.savefig(dir)
        if show:
            plt.show()
        plt.close()
    
    plot_main(frozen_lake, V, pi, cost, np.array(mask))

def custom_grid_search(paramater, learning_func, values, output_type):
    
    print(size)
    np.random.seed(10)
    frozen_lake = gym.make('FrozenLake-v1', render_mode="rgb_array" ,desc=generate_random_map(size=size), is_slippery=slippery, max_episode_steps=max_episode_steps) # gym.make('FrozenLake8x8-v1', render_mode="rgb_array", is_slippery=False)
    frozen_lake = CustomRewardWrapper(frozen_lake, cost, reward)
    output = []
    
   
    
    for val in values:
        frozen_lake.reset()
        
        old = time.time()
        
        if paramater == "gamma":
            if learning_func == "v":
                plan = Planner(frozen_lake.P)
                V, V_track, pi = plan.value_iteration(gamma=val, n_iters=n_iters, theta=theta)
            elif learning_func == "p":
                plan = Planner(frozen_lake.P)
                V, V_track, pi = plan.policy_iteration(gamma=val, n_iters=n_iters, theta=theta)
                
        elif paramater == "n_iters":
            if learning_func == "v":
                plan = Planner(frozen_lake.P)
                V, V_track, pi = plan.value_iteration(gamma=gamma, n_iters=val, theta=theta)
            elif learning_func == "p":
                plan = Planner(frozen_lake.P)
                V, V_track, pi = plan.policy_iteration(gamma=gamma, n_iters=val, theta=theta)
                
        elif paramater == "theta":
            if learning_func == "v":
                plan = Planner(frozen_lake.P)
                V, V_track, pi = plan.value_iteration(gamma=gamma, n_iters=n_iters, theta=val)
            elif learning_func == "p":
                plan = Planner(frozen_lake.P)
                V, V_track, pi = plan.policy_iteration(gamma=gamma, n_iters=n_iters, theta=val)     
                
        new = time.time()
        wall_time = new - old

                
        if(output_type == "mean test score"):
            test_score = TestEnv.test_env(frozen_lake, desc=None, render=False, n_iters=1, pi=pi, user_input=False)
            test_score = round(np.mean(test_score), 2)
            output.append(test_score)
        elif(output_type == "V mean"):
            output.append(np.mean(V))
        elif(output_type) == "time":
            output.append(wall_time)
    return output

# def reward_custom_grid_search(paramater, learning_func, values, output_type):
    
#     np.random.seed(10)
#     output = []
    
    
#     for val in values:
#         frozen_lake = gym.make('FrozenLake-v1', render_mode="rgb_array" ,desc=generate_random_map(size=size), is_slippery=slippery, max_episode_steps=max_episode_steps) # gym.make('FrozenLake8x8-v1', render_mode="rgb_array", is_slippery=False)
#         if paramater == "cost":   
#             frozen_lake = CustomRewardWrapper(frozen_lake, val, reward)
#         elif paramater == "reward":
#             frozen_lake = CustomRewardWrapper(frozen_lake, cost, val)

#         if learning_func == "v":
#             plan = Planner(frozen_lake.P)
#             V, V_track, pi = plan.value_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
#         elif learning_func == "p":
#             plan = Planner(frozen_lake.P)
#             V, V_track, pi = plan.policy_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
                  
                
#         if(output_type == "mean test score"):
#             test_score = TestEnv.test_env(frozen_lake, desc=None, render=False, n_iters=100, pi=pi, user_input=False)
#             test_score = round(np.mean(test_score), 2)
#             output.append(test_score)
#         elif(output_type == "V mean"):
#             output.append(np.mean(V))
#     return output

    


size = 5 # size 5 is that L shape one   #5, 25, 5O
cost = 0
reward = 1
gamma = .9999
n_iters = 100
theta = 1e-150
slippery = True
max_episode_steps = 25000



run_frozen_lake('v', "large_size_theta_issue", show=True)


nums = np.arange(6, 2000, 1)


# gammas = [.01, .99]
# for i, g in enumerate(gammas):
#     gamma = g
#     size = 5
#     cost = -1/(5*5) 
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
    
# for i, g in enumerate(gammas):
#     gamma = g
#     size = 5
    
#     v = custom_grid_search("n_iters", "v", nums, "time")
#     p = custom_grid_search("n_iters", "p", nums, "time")
#     plt.plot(nums,v,
#                     color="b", label = "vi small")
#     plt.plot(nums,p,
#                     color="aqua", label = "pi small")

#     size = 25

#     v = custom_grid_search("n_iters", "v", nums, "time")
#     p = custom_grid_search("n_iters", "p", nums, "time")
#     plt.plot(nums,v,
#                     color="green", label = "vi medium")
#     plt.plot(nums,p,
#                     color="lime", label = "pi medium")

#     size = 50

#     v = custom_grid_search("n_iters", "v", nums, "time")
#     p = custom_grid_search("n_iters", "p", nums, "time")
#     plt.plot(nums,v,
#                     color="r", label = "vi large")
#     plt.plot(nums,p,
#                     color="coral", label = "pi large")

#     plt.title("Iterations v Wall Clock Time: Not slippery and Gamma: " + str(gamma))
#     plt.xlabel("Iterations")
#     plt.ylabel("Wall Clock Time")
#     plt.grid()
#     plt.tight_layout()
#     plt.legend(loc='best')
#     if i == 0:
#         named_size = "small_gamma"
#     else: 
#         named_size = "large_gamma"
#     # plt.savefig("graphs/frozen_lake/" + named_size +"_no_slip_time")
#     plt.close()
    
    
    
# plt.show()


# nums = [.01, .1, .2, .3, .4, .5, .6, .7 ,.8 ,.9, .99,.999]

# v = custom_grid_search("gamma", "v", nums, "mean v score")
# p = custom_grid_search("gamma", "p", nums, "mean v score")
# plt.plot(nums,v,
#                  color="b", label = "vi")
# plt.plot(nums,p,
#                  color="r", label = "pi")

# plt.title("Gamma v Wall Clock Time")
# plt.xlabel("Gamma")
# plt.ylabel("Time")
# plt.grid()
# plt.tight_layout()
# plt.legend(loc='best')
# plt.savefig("graphs/frozen_lake/time_large_slip_gamma")
# plt.show()
