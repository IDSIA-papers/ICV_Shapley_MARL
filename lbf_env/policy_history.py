import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import copy
import random
from lbf_env import LBFEnv
from model.lbf_net import LBFModel
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from scipy.spatial.distance import jensenshannon

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=False)

"""
File for plotting the HISTORY of values, entropy and choices of the LBF environment. NO ICV, yet.

You can manually specify which values to plot by changing the `EXCLUDE_PLOT` variable.

The pre-trained model is NOT perfect, and it sometimes happens that 
some agents keep toggling between two actions / states (you notice in the history plot).
In that case, increase ITERS or perform new runs. 
"""

MODE = 0  # 0: absolute values, 1: differences between two consecutive steps
ITERS = 1

DIR = os.path.dirname(__file__)
ACTS = {0: "0:NONE", 1: "1:UP", 2:"2:DOWN", 3:"3:LEFT", 4:"4:RIGHT", 5:"5:LOAD"}
GAME= "LBF"
LEVEL = 3
HORIZON = 50
env_config = {
    "level":LEVEL,
    "players":3,
    "size":9,
    "max_num_food":3,
    "coop": True,
    "horizon":HORIZON,
    "rew_pot":False,
    "load_punish":False,
    "render":None
}
env = LBFEnv(env_config)

AGENTS = list(range(env.num_agents))
EXCLUDE_PLOT = ["p", "js", "jo"]  # ignore these in plots

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
    policies = {k:algo.get_policy(f"p_{k}") for k in AGENTS}
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
    Centralized observation for Q-value computation without specific action (zero).
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

def cartesian(agent_list, probs):
    """
    Compute joint probabilities and action combinations for list of agents.
    """
    prob_lists = [probs[i] for i in agent_list]
    cartesian_probs = list(itertools.product(*prob_lists))
    joint_probs = [np.prod(prob_tuple) for prob_tuple in cartesian_probs]
    act_ranges = [range(0, len(probs[i])) for i in agent_list]
    action_combinations = list(itertools.product(*act_ranges))
    return joint_probs, action_combinations

def get_V(probs, state, scale=4, shift=0.0):
    """
    Computes the value for each agent in the state based on the joint action probabilities and policies.
    """
    coal = list(AGENTS)
    vals = {k:0.0 for k in coal}
    marginal_probs, marginal_actions = cartesian(coal, probs)
    for m_prob, m_act in zip(marginal_probs, marginal_actions):
        joint_action = {}
        for idx, agent in enumerate(coal):
            joint_action[agent] = m_act[idx]
        for c in coal:
            pol_c = policies[c]
            q_val = pol_c.model.get_value(cc_obs_q_val(state, joint_action, c))
            vals[c] += m_prob * q_val.detach().item() * scale
    if shift>0:
        for k,v in vals.items():
            vals[k] = v + shift
    return vals

def get_I(probs, scale=math.log2(6)):
    """
    Compute entropy and peakedness of the action probabilities for a given state.
    """
    H={}
    P={}
    for i in AGENTS:
        H_max = math.log2(len(probs[i]))
        H[i] = -sum(p * math.log2(p) for p in probs[i] if p > 0) / scale
        P[i] = (H_max-(H[i]*scale)) / scale
    return H, P

def get_jsd(state, scale=True):
    """
    Computes the JSD-based similarity measures:
    - js: if others would act similarly in the state of agent i
    - jo: if agent i would act similarly in the state of others
    """
    jsd = {}
    for i in range(env.num_agents):
        js = 0
        jo = 0
        _, _, info_i = policies[i].compute_single_action(obs=cc_obs(state, i), explore=False)
        probs_i_in_i = torch.softmax(torch.tensor(info_i["action_dist_inputs"]), dim=0).tolist()

        for j in range(env.num_agents):
            if i != j:
                # js component:
                _, _, info_j_in_i = policies[j].compute_single_action(obs=cc_obs(state, i), explore=False)
                probs_j_in_i = torch.softmax(torch.tensor(info_j_in_i["action_dist_inputs"]), dim=0).tolist()
                js += 1 - jensenshannon(probs_i_in_i, probs_j_in_i, 2)

                # jo component:
                _, _, info_i_in_j = policies[i].compute_single_action(obs=cc_obs(state, j), explore=False)
                probs_i_in_j = torch.softmax(torch.tensor(info_i_in_j["action_dist_inputs"]), dim=0).tolist()

                _, _, info_j = policies[j].compute_single_action(obs=cc_obs(state, j), explore=False)
                probs_j_in_j = torch.softmax(torch.tensor(info_j["action_dist_inputs"]), dim=0).tolist()

                jo += 1 - jensenshannon(probs_i_in_j, probs_j_in_j, 2)

        if scale:
            jsd[i] = [js / 2, jo / 2]
        else:
            jsd[i] = [js, jo]
    return jsd

def get_acts_probs(state):
    acts = {}
    probs = {}
    for i in AGENTS:
        acts[i],_,info = policies[i].compute_single_action(obs=cc_obs(state, i), explore=False)
        probs[i] = torch.softmax(torch.tensor(info["action_dist_inputs"]), dim=0).tolist()
    return acts, probs

def plot_history(steps, name="", **kwargs):
    """
    Plot history of player values.
    """
    plt.figure()
    fig, axes = plt.subplots(env.num_agents, 1, figsize=(10, 5), sharey=True)
    colors = ["black", "green", "orange", "red", "dodgerblue", "cyan"]

    for k, ax in enumerate(axes):
        kol = 0
        for kv, vv in kwargs.items():
            if kv not in EXCLUDE_PLOT:
                ax.plot(steps, vv[k], label=f"{name}_{kv}" if k==0 else None, color=colors[kol])
                ax.set_title(f'Player {k}')
                kol+=1
    
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "results_store", f"INFERENCE_history_{name}.png"))
    plt.close()

def more_choices(state, acts, value, scale=4):
    """
    Count actions with higher Q-value than the current action for each agent.
    """
    choices = {k:0 for k in AGENTS}
    for ag_id in AGENTS:
        joint_action = copy.deepcopy(acts)
        for act in ACTS.keys():
            joint_action[ag_id] = act
            q_val = policies[ag_id].model.get_value(cc_obs_q_val(state, joint_action, ag_id))
            q_val = q_val.detach().item() * scale
            if q_val > value[ag_id]:
                choices[ag_id] += 1/len(ACTS)
    return choices

def write_data(steps, name="", **kwargs):
    """
    Stores values from kwargs and writes them to a .dat file.
    
    The output file will have:
      - First row: column headers such as "v0 v1 v2 h0 h1 h2 p0 p1 p2 jp0 jp1 jp2 jn0 jn1 jn2"
      - Second row: corresponding mean values computed from the lists.
    """
    filename = os.path.join(DIR, "results_store", f"lbf_data_{name}.dat")
    with open(filename, "w") as file:
        ks = ""
        for k_name in kwargs.keys(): #val, jp
            for k_agent in kwargs[k_name].keys():#0,1,2
                ks += f"{k_name}{k_agent} " #val0, val1, val2
        file.write(f"x {ks}\n")
        for s in steps:
            s_data = f"{s} "
            for v_name in kwargs.values(): #val={0:[...], 1:[...]}
                for v_agent in v_name.values(): #0:[...]
                     s_data += f"{v_agent[s]} "
            file.write(s_data+"\n")

def avg_data(data):
    """
    Compute the average values for each agent across all timesteps.
    """
    avg_data_val = {k:[] for k in AGENTS}
    for v_s in data.values(): #timestep values
        for k, v_k in v_s.items():
            if len(v_k)>0:
                avg_data_val[k].append(sum(v_k) / len(v_k))
            else:
                avg_data_val[k].append(0)
    return avg_data_val

algo = get_algo()
policies = get_policies(algo)
os.system('clear') if os.name == 'posix' else os.system('cls')

#metrics store either abs values or abs differences during raw inference (no contributions of other players)
metrics = {
    "v": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "h": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "p": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "c": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "js": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "jo": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
}

steps = set()

for it in range(ITERS):
    print(f"iteration {it}")
    state, _ = env.reset(seed=random.randint(0, 100))
    done = False
    s = 0

    while not done:
        acts, probs = get_acts_probs(state)
        values = get_V(probs, state)
        entropies, peaks = get_I(probs)
        choices = more_choices(state, acts, values)
        jsds = get_jsd(state)

        #absolute values
        if MODE == 0:
            for ag_id in AGENTS:
                metrics["v"][s][ag_id].append(values[ag_id])
                metrics["h"][s][ag_id].append(entropies[ag_id])
                metrics["p"][s][ag_id].append(peaks[ag_id])
                metrics["c"][s][ag_id].append(choices[ag_id])
                metrics["js"][s][ag_id].append(jsds[ag_id][0])
                metrics["jo"][s][ag_id].append(jsds[ag_id][1])
            state,_,d,_, _ = env.step(acts)

        #differences
        elif MODE == 1:
            state,_,d,_, _ = env.step(acts)

            acts_new, probs_new = get_acts_probs(state)
            values_new = get_V(probs_new, state)
            entropies_new, peaks_new = get_I(probs_new)
            choices_new = more_choices(state, acts_new, values_new)
            jsds_new = get_jsd(state)

            for ag_id in AGENTS:
                metrics["v"][s][ag_id].append(values_new[ag_id]-values[ag_id])
                metrics["h"][s][ag_id].append(entropies_new[ag_id]-entropies[ag_id])
                metrics["p"][s][ag_id].append(peaks_new[ag_id]-peaks[ag_id])
                metrics["c"][s][ag_id].append(choices_new[ag_id]-choices[ag_id])
                metrics["js"][s][ag_id].append(jsds_new[ag_id][0]-jsds[ag_id][0])
                metrics["jo"][s][ag_id].append(jsds_new[ag_id][1]-jsds[ag_id][1])

        done = d["__all__"]
        steps.add(s)
        s+=1

avg_metrics = {}
for k, v in metrics.items():
    avg_metrics[f"{k}"] = avg_data(v)

plot_history(steps=list(steps), name="abs" if MODE == 0 else "diff", **avg_metrics)
write_data(list(steps), name="abs" if MODE == 0 else "diff", **avg_metrics)

print(f"Finished!")