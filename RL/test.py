import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# # x = gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
# # make gym environment 
# frozen_lake = gym.make('FrozenLake8x8-v1', render_mode="rgb_array", is_slippery=True)
# # frozen_lake = x

# plan = Planner(frozen_lake.P)
# V, V_track, pi = plan.value_iteration(gamma=1, n_iters=1000, theta=1e-10)

# print(frozen_lake.P)

# def get_policy_map(pi, val_max, actions, map_size):
#         """Map the best learned action to arrows."""
#         #convert pi to numpy array
#         best_action = np.zeros(val_max.shape[0], dtype=np.int32)
#         for idx, val in enumerate(val_max):
#             best_action[idx] = pi[idx]
#             # print(val)
#             if val == 0.0:
#                 # print("here")
#                 best_action[idx] = 4
#                 pi[idx] = 4
#         print(best_action)
#         policy_map = np.empty(best_action.flatten().shape, dtype=str)
#         for idx, val in enumerate(best_action.flatten()):
            
#             policy_map[idx] = actions[val]
#         policy_map = policy_map.reshape(map_size[0], map_size[1])
#         val_max = val_max.reshape(map_size[0], map_size[1])
#         return val_max, policy_map


# def plot_problem(env):

#     frozen_lake.reset()
#     plt.imshow(frozen_lake.render())
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
# def plot_policy(val_max, directions, map_size, title):
#         """Plot the policy learned."""
    
            
        
#         sns.heatmap(
#             val_max,
#             annot=directions,
#             fmt="",
        
#             cmap=sns.color_palette("Blues", as_cmap=True),
#             linewidths=0.7,
#             linecolor="black",
#             xticklabels=[],
#             yticklabels=[],
#             annot_kws={"fontsize": "xx-large"},
#         ).set(title=title)
#         img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
#         plt.show()



# #plot state values
# map_size=(8,8)

# fl_actions = {0: "←", 1: "↓", 2: "→", 3: "↑", 4:""}
# title="FL Mapped Policy\nArrows represent best action"
# val_max, policy_map = get_policy_map(pi, V, fl_actions, map_size)

# print(policy_map)
# # Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", map_size)
# # Plots.v_iters_plot(V, "Frozen Lake\nValue Iteration State Values")
# plot_policy(val_max, policy_map, map_size, title)
# plot_problem(frozen_lake)




# def plot_q_values_map(policy_map, env, map_size):
#     """Plot the last frame of the simulation and the policy learned."""
#     env.reset()

#     # Plot the last frame
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
#     ax[0].imshow(env.render())
#     ax[0].axis("off")
#     ax[0].set_title("Last frame")

#     # Plot the policy
#     sns.heatmap(
#         val_max,
#         annot=policy_map,
#         fmt="",
#         ax=ax[1],
#         cmap=sns.color_palette("Blues", as_cmap=True),
#         linewidths=0.7,
#         linecolor="black",
#         xticklabels=[],
#         yticklabels=[],
#         annot_kws={"fontsize": "xx-large"},
#     ).set(title="Learned Q-values\nArrows represent best action")
#     # for _, spine in ax[1].spines.items():
#     #     spine.set_visible(True)
#     #     spine.set_linewidth(0.7)
#     #     spine.set_color("black")
#     # img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
#     # fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
#     plt.show()


# # plot_q_values_map(policy_map, frozen_lake, map_size)



import numpy as np
import seaborn as sns
import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def adjust_render(render, map_size):
    pixels = render.shape[0] // map_size * map_size
    return render[:pixels, :pixels, :]

map_size = 10
env = gym.make(
    "FrozenLake-v1",
    desc=generate_random_map(size=map_size),
    is_slippery=True,
    render_mode="rgb_array",
)
env.reset()

V, V_track, pi = Planner(env.P).value_iteration()
print(V_track)




# pi = [pi[key] for key in range(env.observation_space.n)]

# # Plot the policy
# reshaped_V = V.reshape(map_size, map_size)
# directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
# hmax = sns.heatmap(
#     reshaped_V,
#     annot=np.reshape([directions[val] for val in pi], (map_size, map_size)),
#     fmt="",
#     cmap=sns.color_palette("Blues", as_cmap=True),
#     linewidths=0.7,
#     linecolor="black",
#     xticklabels=[],
#     yticklabels=[],
#     annot_kws={"fontsize": "xx-large"},
#     mask=reshaped_V < np.finfo(float).eps, # Minimum float number on the machine
#     zorder = 2,
# )
# hmax.set(title="Optimal Policy")
# hmax.imshow(adjust_render(env.render(), map_size),
#             aspect = hmax.get_aspect(),
#             extent = hmax.get_xlim() + hmax.get_ylim(),
#             zorder = 1)
# plt.show()