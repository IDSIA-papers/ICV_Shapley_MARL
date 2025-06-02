import os
import matplotlib.pyplot as plt
import torch
from lbf_env import LBFEnv
import ray
from ray.rllib.models import ModelCatalog
from model.lbf_net import LBFModel
from ray.rllib.algorithms.algorithm import Algorithm

"""
File for plotting the policy distribution and value of the LBF environment. NO ICV, yet.
The user can manually input actions for each agent.
"""

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=False)

ITERS = 5
ACTS = {0: "0:NONE", 1: "1:UP", 2:"2:DOWN", 3:"3:LEFT", 4:"4:RIGHT", 5:"5:LOAD"}
DIR = os.path.dirname(__file__)
GAME= "LBF"
LEVEL = 3
HORIZON = 50
env_config = {
    "level":LEVEL,
    "players":3,
    "size":9,
    "max_num_food":3,
    "coop": True,
    "horizon":50,
    "rew_pot":False,
    "load_punish":False,
    "render":None
}
env = LBFEnv(env_config)

store_path = os.path.join(DIR, "results_store")
if not os.path.exists(store_path):
    os.makedirs(store_path)

def get_algo():
    ModelCatalog.register_custom_model("lbf_model", LBFModel)
    rest_path = os.path.join(DIR, "model", "checkpoint")
    algo = Algorithm.from_checkpoint(rest_path)
    algo.config["num_rollout_workers"] = 0
    algo.config["num_envs_per_worker"] = 1
    algo.config["create_env_on_driver"] = False
    algo.config["explore"] = False
    algo.config["evaluation_interval"] = None
    algo.config["disable_env_checking"] = True 
    algo.config["disable_execution_plan_api"] = True
    return algo

def get_policies(algo):
    policies = {k:algo.get_policy(f"p_{k}") for k in range(env.num_agents)}
    return policies

def cc_obs_q_val(obs, act,i):
    """
    Centralized observation for Q-value computation with specific action.
    """
    if i ==0:
        r=[0,1,2]
    elif i==1:
        r=[1,0,2]
    elif i==2:
        r=[2,0,1]
    return {
        "obs_1_own": torch.tensor(obs[r[0]]).unsqueeze(0),
        "obs_2": torch.tensor(obs[r[1]]).unsqueeze(0),
        "obs_3": torch.tensor(obs[r[2]]).unsqueeze(0),
        "act_1_own": torch.tensor([[act[r[0]]]]),
        "act_2": torch.tensor([[act[r[1]]]]),
        "act_3": torch.tensor([[act[r[2]]]]),
    }

def cc_obs(obs,i):
    """
    Centralized observation for Q-value computation without action (zero).
    """
    if i ==0:
        r=[0,1,2]
    elif i==1:
        r=[1,0,2]
    elif i==2:
        r=[2,0,1]
    return {
        "obs_1_own": torch.tensor(obs[r[0]]),
        "obs_2": torch.tensor(obs[r[1]]),
        "obs_3": torch.tensor(obs[r[2]]),
        "act_1_own": torch.tensor([0]),
        "act_2": torch.tensor([0]),
        "act_3": torch.tensor([0]),
    }

def plot_dis(logits, value):
    """
    Plot the policy distribution and value for each agent.
    """
    fig, axes = plt.subplots(1, env.num_agents, figsize=(12, 4), sharey=True)

    for k, ax in enumerate(axes):
        probs = torch.softmax(logits[k], dim=0).tolist()
        ax.bar(list(ACTS.keys()), probs, tick_label=list(ACTS.values()), label="Probs" if k==0 else None)
        ax.axhline(y=value[k], color='red', linestyle='-', label=f'Value' if k==0 else None)

        ax.set_xlabel('Actions')
        ax.set_title(f'Player {k}')
    
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    fig.text(0.0, 0.5, 'Values', va='center', rotation='vertical')
    plt.tight_layout()
    #plt.ylabel('Prob. / Value')
    #plt.title('Policy Distribution & Value')
    plt.savefig(os.path.join(DIR, "results_store", f"DISTRIBUTION_{GAME}.png"))

algo = get_algo()
policies = get_policies(algo)

os.system('clear') if os.name == 'posix' else os.system('cls')

for _ in range(ITERS):
    state, _ = env.reset()
    reward =0
    done = False
    acts = {}
    logits = {}
    values = {}

    while not done:
        env.render()
        for i in range(env.num_agents):
            acts[i],_,info = policies[i].compute_single_action(obs=cc_obs(state, i), explore=False)
            logits[i] = torch.tensor(info["action_dist_inputs"])
        
        for i in range(env.num_agents):
            val = policies[i].model.get_value(cc_obs_q_val(state, acts,i))
            values[i] = round(val.detach().item(), 4)

        plot_dis(logits, values)

        a = input(f"chosen policy actions={[ACTS[j] for j in acts.values()]} ---> manual input (integers): ")
        if a=="k":
            done=True
        else:
            act_manual = {i:int(a) for i,a in enumerate(a)}

            state,r,d,_, _ = env.step(acts if len(a)==1 else act_manual)
            done = d["__all__"]