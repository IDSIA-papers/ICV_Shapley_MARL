
import os
import torch
import time
import matplotlib.pyplot as plt
from spread_tag_env import TagEnv
import ray
from ray.rllib.models import ModelCatalog
from model.mpe_net import TagNetAgentSymmIndep
from ray.rllib.algorithms.algorithm import Algorithm

"""
File for plotting the policy distribution and value of the MPE Tag environment. NO ICV, yet.
The user can manually input actions for each agent.
"""

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=False)

ITERS = 5
ACTS = {0: "0:NOP", 1:"1:LEFT", 2:"2:RIGHT", 3:"3:DOWN", 4:"4:UP"}
DIR = os.path.dirname(__file__)
GAME= "MPE"
N_GOOD= 2
N_ADV = 2
N_OB = 0
env_config={
    "num_good": N_GOOD,
    "num_adv": N_ADV,
    "num_obst": N_OB,
    "max_cylces": 50,
    "continuous_actions": False,
    "render_mode": "human",
    "seed": None
}
agents = ['adversary_0', 'adversary_1', 'agent_0', 'agent_1']
env = TagEnv(env_config)

store_path = os.path.join(DIR, "results_store")
if not os.path.exists(store_path):
    os.makedirs(store_path)

def get_algo():
    ModelCatalog.register_custom_model("tag_model_agent", TagNetAgentSymmIndep)
    rest_path = os.path.join(DIR, "model", f"tag_e001", "checkpoint")
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
    policies = {k:algo.get_policy(k) for k in agents}
    return policies

def plot_dis(logits, value):
    """
    Plot the policy distribution and value for each agent.
    """
    fig, axes = plt.subplots(1, env.num_agents, figsize=(15, 4), sharey=True)

    for k, ax in enumerate(axes):
        probs = torch.softmax(logits[k], dim=0).tolist()
        ax.bar(list(ACTS.keys()), probs, tick_label=list(ACTS.values()), label="Probs" if k==0 else None)
        ax.axhline(y=value[k], color='red', linestyle='-', label=f'Value' if k==0 else None)

        ax.set_xlabel('Actions')
        ax.set_title(f'{agents[k]}')
    
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    fig.text(0.0, 0.5, 'Values', va='center', rotation='vertical')
    plt.tight_layout()
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
        for i, ag_id in enumerate(agents):
            acts[ag_id],_,info = policies[ag_id].compute_single_action(state[ag_id], explore=False)
            logits[i] = torch.tensor(info["action_dist_inputs"])
            values[i] = info["vf_preds"]

        plot_dis(logits, values)

        a = input(f"chosen policy actions={[ACTS[j] for j in acts.values()]} ---> manual input (integers): ")
        if a=="k":
            done=True
        else:
            act_manual = {ag_id:int(a) for ag_id,a in zip(agents, a)}

            state,r,d,_, _ = env.step(acts if len(a)==1 else act_manual)
            done = d["__all__"]