import os
import torch
import random
import math
import ray
import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt
from lbf_env import LBFEnv
from model.lbf_net import LBFModel
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from scipy.spatial.distance import jensenshannon

"""
File for plotting ICVs in the LBF environment.
Contributions of each player to the others are computed, as well as absolute values of each player.
"""

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=False)

ITERS = 500
WITH_ABS = False  # if True, also compute absolute values of each player

ACTS = {0: "0:NONE", 1: "1:UP", 2:"2:DOWN", 3:"3:LEFT", 4:"4:RIGHT", 5:"5:LOAD"}
N_ACTS = len(ACTS)
DIR = os.path.dirname(__file__)
GAME= "LBF"
LEVEL = 3
HORIZON = 30
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
EXCLUDE_PLOT = ["h"]  # ignore these in plots

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
                choices[ag_id] += 1/N_ACTS
    return choices

def get_V(probs, state, scale=4, shift=0.0):
    """
    Computes the value for each agent in the state based on the joint action probabilities and policies.
    """
    coal = list(range(env.num_agents))
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
    for i in range(env.num_agents):
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
    for i in range(env.num_agents):
        acts[i],_,info = policies[i].compute_single_action(obs=cc_obs(state, i), explore=False)
        probs[i] = torch.softmax(torch.tensor(info["action_dist_inputs"]), dim=0).tolist()
    return acts, probs

def avg_data(data):
    """
    Compute the average values for each agent across all timesteps.
    """
    avg_data = {k:[] for k in range(env.num_agents)}
    for v_s in data.values(): #timestep values
        for k, v_k in v_s.items():
            if len(v_k)>0:
                avg_data[k].append(sum(v_k) / len(v_k))
            else:
                avg_data[k].append(0)
    return avg_data

def write_data_history(steps, name="", **kwargs):
    """"
    Store history data from kwargs and write it to a .dat file.
    """
    filename= os.path.join(DIR, "results_store", f"lbf_data_history{name}.dat")
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

def write_data_icv(**kwargs):
    """
    Stores mean values from kwargs and writes them to a .dat file.
    
    The output file will have:
      - First row: column headers formatted as: value_acting-agent-idx such as v_1
      - Second row: corresponding mean ICV values.
    """
    filename= os.path.join(DIR, "results_store", "lbf_data_icv.dat")
    measure_map = {
        "avg_v": "v",
        "avg_h": "h",
        "avg_p": "p",
        "avg_jp": "jp",
        "avg_jn": "jn",
        "avg_c": "c"
    }
    
    categories = []
    means = []
    
    for measure in measure_map:
        if measure not in kwargs:
            continue
        measure_short = measure_map[measure]

        for key in sorted(kwargs[measure].keys()):
            cat_name = f"{measure_short}{key}"
            categories.append(cat_name)
            data = kwargs[measure][key]
            mean_val = np.mean(data) if data else 0
            means.append(mean_val)
    
    with open(filename, "w") as f:
        f.write(" ".join(categories) + "\n")
        f.write(" ".join(str(val) for val in means) + "\n")

def plot_history(steps, name="", **kwargs):
    """
    Plot ICV history of player effects.
    """
    plt.figure()
    fig, axes = plt.subplots(env.num_agents, 1, figsize=(10, 5), sharey=True)
    colors = ["black", "green", "orange", "red", "dodgerblue", "cyan"]

    for k, ax in enumerate(axes):
        kol = 0
        for kv, vv in kwargs.items():
            lab = kv.split('_')[1]
            if lab not in EXCLUDE_PLOT or name == "_abs":
                ax.plot(steps, vv[k], label=lab if k==0 else None, color=colors[kol])
                ax.set_title(f'Player {k}')
                kol+=1
    
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "results_store", f"ICV_history{name}.png"))
    plt.close()

def plot_icv(**kwargs):
    """
    Plot the ICV bars for each player.
    """
    colors = ["black", "green", "orange", "red", "dodgerblue", "cyan"]
    for e in EXCLUDE_PLOT:
        ek = f"avg_{e}"
        if ek in kwargs:
            del kwargs[ek]
    plt.figure()
    metrics = list(kwargs.keys())
    x_vals = sorted(kwargs[metrics[0]].keys())
    
    means = {m: np.array([np.mean(kwargs[m][x]) for x in x_vals]) 
             for m in metrics}
    
    bar_width = 0.15
    n_metrics = len(metrics)
    # The offset for the k-th bar is (k - (n-1)/2) * bar_width.
    offsets = [ (i - (n_metrics - 1) / 2) * bar_width for i in range(n_metrics)]
    xs = np.array(x_vals)
    
    # Plot each metric as a set of bars with a corresponding offset.
    for i, metric in enumerate(metrics):
        plt.bar(xs + offsets[i], means[metric], width=bar_width, label=metric, color=colors[i])

    plt.xlabel('Player')
    plt.ylabel('Mean Value')
    plt.title('ICV')
    plt.xticks(xs)
    plt.legend()
    plt.savefig(os.path.join(DIR, "results_store", f"ICV.png"))
    plt.close()

algo = get_algo()
policies = get_policies(algo)
os.system('clear') if os.name == 'posix' else os.system('cls')

steps = set()

# metrics store contributions of each player to the others
metrics = {
    "diff_v": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_h": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_p": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_js": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_jo": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_c": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
}
# metrics_abs store absolute values of each player
metrics_abs = {
    "abs_v": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "abs_h": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "abs_c": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
}

print("Start game and ICV computation...")

for it in range(ITERS):
    print(f"iteration {it}")
    #state, _ = env.reset(seed=it*50 + random.randint(0, 50))
    state, _ = env.reset(seed=random.randint(0, 100))
    done = False
    s = 0
    acting_agents = AGENTS.copy()

    while not done:
        acts, probs = get_acts_probs(state)

        affected_agents = AGENTS.copy()
        # Sample a random order of acting agents (sigma)
        random.shuffle(acting_agents)

        for acting_id in acting_agents:
            affected_agents.remove(acting_id)
            values = get_V(probs, state)
            entropies, peaks = get_I(probs)
            jsds = get_jsd(state)
            choices = more_choices(state, acts, values)

            joint_act = {ag:0 for ag in AGENTS}
            joint_act[acting_id] = acts[acting_id]

            state, r, d, _, _ = env.step(action_dict=joint_act,count=len(affected_agents)==0)
            done = d["__all__"]

            acts_new, probs_new = get_acts_probs(state)
            values_new = get_V(probs_new, state)
            jsds_new = get_jsd(state)
            entropies_new, peaks_new = get_I(probs_new)
            choices_new = more_choices(state, acts_new, values_new)

            for affected_id in affected_agents:
                metrics["diff_v"][s][acting_id].append(values_new[affected_id]-values[affected_id])
                metrics["diff_h"][s][acting_id].append(entropies_new[affected_id]-entropies[affected_id])
                metrics["diff_p"][s][acting_id].append(peaks_new[affected_id]-peaks[affected_id])
                metrics["diff_js"][s][acting_id].append(jsds_new[affected_id][0]-jsds[affected_id][0])
                metrics["diff_jo"][s][acting_id].append(jsds_new[affected_id][1]-jsds[affected_id][1])
                metrics["diff_c"][s][acting_id].append(choices_new[affected_id]-choices[affected_id])

            if WITH_ABS:
                metrics_abs["abs_v"][s][acting_id].append(values[acting_id])
                metrics_abs["abs_h"][s][acting_id].append(entropies[acting_id])
                metrics_abs["abs_c"][s][acting_id].append(choices[acting_id])

        steps.add(s)
        s+=1

# average the metrics across all timesteps
avg_metrics = {}
for k,v in metrics.items():
    met = k.split("_")[1]
    avg_metrics[f"avg_{met}"] = avg_data(v)

# Plot and store of metrics
plot_history(list(steps), **avg_metrics)
write_data_history(list(steps), **avg_metrics)

plot_icv(**avg_metrics)
write_data_icv(**avg_metrics)

if WITH_ABS:
    avg_metrics_abs = {}
    for k,v in metrics_abs.items():
        met = k.split("_")[1]
        avg_metrics_abs[f"avg_{met}"] = avg_data(v)

    plot_history(list(steps), name="_abs", **avg_metrics_abs)
    write_data_history(list(steps), name="_abs", **avg_metrics_abs)
    
print("finish")
