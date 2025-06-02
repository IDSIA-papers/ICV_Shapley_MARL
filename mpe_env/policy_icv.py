import matplotlib.pyplot as plt
import os
import torch
import math
import numpy as np
import random, copy
from scipy.spatial.distance import jensenshannon
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from spread_tag_env import TagEnv
from model.mpe_net import TagNetAgentSymmIndep

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=False)

"""
File for plotting ICV measures in the MPE Tag environment.
Contributions of each player to the others are computed, as well as absolute values of each player.
"""

ITERS = 500
WITH_ABS = False  # if True, also compute absolute values of each player

ACTS = {0: "0:NOP", 1:"1:LEFT", 2:"2:RIGHT", 3:"3:DOWN", 4:"4:UP"}
AGENT_NAMES = {0:'adversary_0', 1:'adversary_1', 2:'agent_0', 3: 'agent_1'}
AGENT_NAMES_SHORT = {0:'adv_0', 1:'adv_1', 2:'ag_0', 3:'ag_1'}
AGENT_KEYS = {'adversary_0':0, 'adversary_1':1, 'agent_0':2, 'agent_1':3}

DIR = os.path.dirname(__file__)
GAME= "MPE"
N_GOOD= 2
N_ADV = 2
N_OB = 0
HORIZON = 50
env_config={
    "num_good": N_GOOD,
    "num_adv": N_ADV,
    "num_obst": N_OB,
    "max_cylces": HORIZON,
    "continuous_actions": False,
    "render_mode": None,
    "seed": None
}
env = TagEnv(env_config)

AGENTS = list(range(env.num_agents))
EXCLUDE_PLOT = ["h"] #ignore these in plots

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
    policies = {k:algo.get_policy(k) for k in AGENT_NAMES.values()}
    return policies

def get_I(probs, scale=math.log2(5)):
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

def get_jsd(state, scale=1):
    """
    Computes the JSD-based similarity measures:
    - jc: similarity of agent i to agent j (consensus)
    - jd: dissimilarity of agent i to agent j (divergence)
    """
    jsd = {}

    for i in AGENTS:
        js_sim = 0  # similarity from j acting in i's state
        jo_sim = 0  # similarity from i acting in j's state
        js_div = 0  # divergence from j acting in i's state
        jo_div = 0  # divergence from i acting in j's state

        _, _, info_i = policies[AGENT_NAMES[i]].compute_single_action(obs=state[AGENT_NAMES[i]], explore=False)
        probs_i_in_i = torch.softmax(torch.tensor(info_i["action_dist_inputs"]), dim=0).tolist()

        for j in AGENTS:
            if i != j:
                # self part
                _, _, info_j_in_i = policies[AGENT_NAMES[j]].compute_single_action(obs=state[AGENT_NAMES[i]], explore=False)
                probs_j_in_i = torch.softmax(torch.tensor(info_j_in_i["action_dist_inputs"]), dim=0).tolist()
                jsd_ji = jensenshannon(probs_i_in_i, probs_j_in_i, base=2)

                # check for same or different type of team-members. 
                if AGENT_NAMES[i].startswith("age"):
                    if AGENT_NAMES[j].startswith("age"): 
                        js_sim += 1 - jsd_ji
                    else:
                        js_div += jsd_ji
                elif AGENT_NAMES[i].startswith("adv"):
                    if AGENT_NAMES[j].startswith("adv"): 
                        js_sim += 1 - jsd_ji
                    else:
                        js_div += jsd_ji

                # other part
                _, _, info_i_in_j = policies[AGENT_NAMES[i]].compute_single_action(obs=state[AGENT_NAMES[j]], explore=False)
                probs_i_in_j = torch.softmax(torch.tensor(info_i_in_j["action_dist_inputs"]), dim=0).tolist()

                _, _, info_j = policies[AGENT_NAMES[j]].compute_single_action(obs=state[AGENT_NAMES[j]], explore=False)
                probs_j_in_j = torch.softmax(torch.tensor(info_j["action_dist_inputs"]), dim=0).tolist()
                jsd_ij = jensenshannon(probs_i_in_j, probs_j_in_j, base=2)
                
                # check for same or different type of team-members. 
                if AGENT_NAMES[i].startswith("age"):
                    if AGENT_NAMES[j].startswith("age"): 
                        jo_sim += 1 - jsd_ij
                    else:
                        jo_div += jsd_ij
                elif AGENT_NAMES[i].startswith("adv"):
                    if AGENT_NAMES[j].startswith("adv"): 
                        jo_sim += 1 - jsd_ij
                    else:
                        jo_div += jsd_ij

        if scale:
            jsd[i] = [(js_sim + jo_sim)/2, (js_div + jo_div) / 4, (js_sim + jo_sim + js_div + jo_div) / 6]
        else:
            jsd[i] = [js_sim + jo_sim, js_div + jo_div, js_sim + jo_sim + js_div + jo_div]

    return jsd

def get_acts_probs_values(state, scale=1):
    acts = {}
    probs = {}
    values = {}
    for i in range(env.num_agents):
        acts[AGENT_NAMES[i]],_,info = policies[AGENT_NAMES[i]].compute_single_action(obs=state[AGENT_NAMES[i]], explore=False)
        probs[i] = torch.softmax(torch.tensor(info["action_dist_inputs"]), dim=0).tolist()
        values[i] = info["vf_preds"]
    return acts, probs, values

def avg_data(data, indiv=False):
    """
    Compute the average values for each agent across all timesteps.
    """
    avg_data = {k:[] for k in range(env.num_agents)}
    avg_data_indiv = {k:{j:[] for j in AGENT_NAMES_SHORT.values()} for k in AGENT_NAMES_SHORT.values()}
    for v_s in data.values(): #timestep values: {0:[...], 1:[...]}
        for k, v_k in v_s.items():  #k=ag_idx, v_k = list values [...]
            if indiv:
                for k_aff, v_aff in v_k.items():
                    if len(v_aff)>0:
                        avg_data_indiv[AGENT_NAMES_SHORT[k]][AGENT_NAMES_SHORT[k_aff]].append(sum(v_aff)/len(v_aff))
                    else:
                        avg_data_indiv[AGENT_NAMES_SHORT[k_aff]][AGENT_NAMES_SHORT[k_aff]].append(0)
            else:
                if len(v_k)>0:
                    avg_data[k].append(sum(v_k) / len(v_k))
                else:
                    avg_data[k].append(0)
    return avg_data if not indiv else avg_data_indiv

def write_data_history(steps, name="", **kwargs):
    """"
    Store history data from kwargs and write it to a .dat file.
    """
    filename= os.path.join(DIR, "results_store", f"mpe_data_history_{name}.dat")
    with open(filename, "w") as file:
        ks = ""
        for k_metric in kwargs.keys(): #avg_v, avg_h,...
            for k_agent in kwargs[k_metric].keys():#0,1,2
                ks += f"{k_metric}{k_agent} " #avg_v0, avg_v1,...
        file.write(f"x {ks}\n")
        for s in steps:
            s_data = f"{s} "
            for v_metric in kwargs.values(): #{0:[...], 1:[...]}
                for v_agent in v_metric.values(): #0:[...]
                     s_data += f"{v_agent[s]} "
            file.write(s_data+"\n")

def write_data_icv(**kwargs):
    """
    Stores mean values from kwargs and writes them to a .dat file.
    
    The output file will have:
      - First row: column headers formatted as: value, acting agent, affected agent, such as v_ad0_ag1
      - Second row: corresponding mean ICV values.
    """
    filename= os.path.join(DIR, "results_store", "mpe_data_icv.dat")
    measure_map = {
        "avg_v": "v",
        "avg_h": "h",
        "avg_p": "p",
        "avg_jp": "jp",
        "avg_jn": "jn"
    }

    agents = ["adv_0", "adv_1", "ag_0", "ag_1"]

    # Helper function to shorten agent names.
    def short_agent(agent):
        if agent.startswith("adv_"):
            return "ad" + agent.split("_")[1]
        elif agent.startswith("ag_"):
            return "ag" + agent.split("_")[1]
        else:
            return agent

    categories = []
    means = []
    
    for measure in measure_map:
        if measure not in kwargs:
            continue  # Skip measures not present in the kwargs
        measure_short = measure_map[measure]
        # Loop over each "acting agent" and each "affected agent"
        for acting in agents:
            for affected in agents:
                cat_name = f"{measure_short}_{short_agent(acting)}_{short_agent(affected)}"
                categories.append(cat_name)
                data = kwargs[measure].get(acting, {}).get(affected, [])
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
    fig, axes = plt.subplots(env.num_agents, 1, figsize=(12, 8), sharey=True)
    colors = ["black", "green", "orange", "red", "dodgerblue", "cyan"]

    for k, ax in enumerate(axes):
        kol = 0
        for kv, vv in kwargs.items():
            if kv not in EXCLUDE_PLOT:
                ax.plot(steps, vv[k], label=f"{name}_{kv}" if k==0 else None, color=colors[kol])
                ax.set_title(AGENT_NAMES[k])
                kol+=1
    
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "results_store", f"ICV_history_{name}.png"))
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
    
    # Calculate mean values for each metric and x value.
    means = {m: np.array([np.mean(kwargs[m][x]) for x in x_vals]) 
             for m in metrics}
    
    # Set bar width and compute offsets for 3 bars per group.
    bar_width = 0.2
    n_metrics = len(metrics)
    # The offset for the k-th bar is (k - (n-1)/2) * bar_width.
    offsets = [ (i - (n_metrics - 1) / 2) * (bar_width) for i in range(n_metrics)]
    
    group_spacing = 1.5
    xs = np.arange(len(x_vals)) * group_spacing
    
    # Plot each metric as a set of bars with a corresponding offset.
    for i, metric in enumerate(metrics):
        plt.bar(xs + offsets[i], means[metric], width=bar_width, label=metric, color=colors[i])

    # Labeling and formatting the plot.
    plt.xlabel('Player')
    plt.ylabel('Mean Value')
    plt.title('ICV')
    plt.xticks(xs)  # Use the x values (e.g., 0, 1, 2) as x-tick labels.
    plt.legend()
    plt.savefig(os.path.join(DIR, "results_store", f"ICV.png"))
    plt.close()

def plot_icv_indiv(**kwargs):
    """
    ICV bars for each player by plotting 4 vertical barplots showing mean values for each measure.
    """
    colors = ["black", "green", "dodgerblue", "cyan"]
    # Define the measures and agent labels in the order we want them to appear.
    measures = ['avg_v', 'avg_p', 'avg_j-team', 'avg_j-opp']
    agents = ['adv_0', 'adv_1', 'ag_0', 'ag_1']
    
    plt.figure()
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 12), constrained_layout=True)
    
    # x-axis positions for the affected agents
    x = np.arange(len(agents))
    bar_width = 0.15
    n_measures = len(measures)
    
    # Loop over each affecting agent (each subplot corresponds to one affecting agent)
    for idx, affecting_agent in enumerate(agents):
        ax = axes[idx]
        
        # For each measure, compute and plot the mean values for each affected agent.
        for m_index, measure in enumerate(measures):
            means = []
            for affected_agent in agents:
                # Get the list for the combination and compute its mean.
                # If the list is empty, default to 0.
                data = kwargs.get(measure, {}).get(affecting_agent, {}).get(affected_agent, [])
                mean_val = np.mean(data) if data else 0
                means.append(mean_val)
            
            # Plot each measure's bar with an offset for grouping.
            ax.bar(x + m_index * bar_width, means, width=bar_width, 
                   label=measure if idx == 0 else "", align='center', color=colors[m_index])
        
        # Center the x-ticks for each group of bars.
        ax.set_xticks(x + (n_measures * bar_width) / 2 - (bar_width / 2))
        ax.set_xticklabels(agents)
        ax.set_title(f"Effect of {affecting_agent} on")
        ax.set_ylabel("Mean Value")
    
    axes[0].legend(loc='upper right')
    # Set an x-label for the bottom subplot.
    #axes[-1].set_xlabel("Affected Agent")

    plt.savefig(os.path.join(DIR, "results_store", f"ICV_indiv.png"))
    plt.close()

def seq_agent_obs(ag_id, states, s):
    """
    Partial observation of the agent. 
    """
    p_vel = states[s+1][ag_id][:2]
    p_pos = states[s+1][ag_id][2:4]
    if ag_id.startswith("agent"):
        oth_pos1 = states[s]["adversary_0"][2:4]
        oth_vel1 = states[s]["adversary_0"][:2]
        oth_pos2 = states[s]["adversary_1"][2:4]
        oth_vel2 = states[s]["adversary_1"][:2]
    elif ag_id.startswith("adversary"):
        oth_pos1 = states[s]["agent_0"][2:4]
        oth_vel1 = states[s]["agent_0"][:2]
        oth_pos2 = states[s]["agent_1"][2:4]
        oth_vel2 = states[s]["agent_1"][:2]
    else:
        raise ValueError(f"agent id {ag_id} not recognized.")
    return np.concatenate(
        [p_vel] + [p_pos] + [oth_pos1-p_pos] + [oth_vel1] + [oth_pos2-p_pos] + [oth_vel2]
    )

def seq_agent_obs_indiv(affected, acting, states, s):
    """
    Returns the observation sequence for the affected agent when the acting agent acts.
    """
    if affected == acting:
        #effect of acting on itself
        return seq_agent_obs(acting, states, s)
    else:
        p_vel = states[s][affected][:2]
        p_pos = states[s][affected][2:4]
        if affected.startswith("agent"):
            if acting.endswith("y_0"):
                oth_pos1 = states[s+1]["adversary_0"][2:4]
                oth_vel1 = states[s+1]["adversary_0"][:2]
                oth_pos2 = states[s]["adversary_1"][2:4]
                oth_vel2 = states[s]["adversary_1"][:2]
            elif acting.endswith("y_1"):
                oth_pos1 = states[s]["adversary_0"][2:4]
                oth_vel1 = states[s]["adversary_0"][:2]
                oth_pos2 = states[s+1]["adversary_1"][2:4]
                oth_vel2 = states[s+1]["adversary_1"][:2]
            else:
                oth_pos1 = states[s]["adversary_0"][2:4]
                oth_vel1 = states[s]["adversary_0"][:2]
                oth_pos2 = states[s]["adversary_1"][2:4]
                oth_vel2 = states[s]["adversary_1"][:2]
        elif affected.startswith("adversary"):
            if acting.endswith("t_0"):
                oth_pos1 = states[s+1]["agent_0"][2:4]
                oth_vel1 = states[s+1]["agent_0"][:2]
                oth_pos2 = states[s]["agent_1"][2:4]
                oth_vel2 = states[s]["agent_1"][:2]
            elif acting.endswith("t_1"):
                oth_pos1 = states[s]["agent_0"][2:4]
                oth_vel1 = states[s]["agent_0"][:2]
                oth_pos2 = states[s+1]["agent_1"][2:4]
                oth_vel2 = states[s+1]["agent_1"][:2]
            else:
                oth_pos1 = states[s]["agent_0"][2:4]
                oth_vel1 = states[s]["agent_0"][:2]
                oth_pos2 = states[s]["agent_1"][2:4]
                oth_vel2 = states[s]["agent_1"][:2]
        else:
            raise ValueError(f"agent id {affected} or {acting} not recognized.")
        return np.concatenate(
            [p_vel] + [p_pos] + [oth_pos1-p_pos] + [oth_vel1] + [oth_pos2-p_pos] + [oth_vel2]
        )

algo = get_algo()
policies = get_policies(algo)
os.system('clear') if os.name == 'posix' else os.system('cls')

steps = set()

# Store metrics for absolute values and differences
metrics_abs = {
    "abs_v": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "abs_p": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "abs_j-team": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "abs_j-opp": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
}
metrics = {
    "diff_v": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_p": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_j-team": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_j-opp": { s:{i:[] for i in range(env.num_agents)} for s in range(HORIZON)},
}

# Store metrics for individual effects of each agent on the others
metrics_indiv = {
    "diff_v": { s:{i:{k:[] for k in range(env.num_agents)} for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_p": { s:{i:{k:[] for k in range(env.num_agents)} for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_j-team": { s:{i:{k:[] for k in range(env.num_agents)} for i in range(env.num_agents)} for s in range(HORIZON)},
    "diff_j-opp": { s:{i:{k:[] for k in range(env.num_agents)} for i in range(env.num_agents)} for s in range(HORIZON)},
}

# Store states for each iteration and timestep
state_store = { it:{s:None for s in range(HORIZON)} for it in range(ITERS)}

print("Start game and ICV computation...")

for it in range(ITERS):
    print(f"iteration {it}")
    state, _ = env.reset(seed=random.randint(0, 100))
    done = False
    s = 0

    while not done:
        state_store[it][s] = state
        acts = {}
        for i in range(env.num_agents):
            acts[AGENT_NAMES[i]],_,info = policies[AGENT_NAMES[i]].compute_single_action(obs=state[AGENT_NAMES[i]], explore=False)
        state, _, term, _, _ = env.step(acts)
        done = term["__all__"]
        if not done:
            steps.add(s)
            s+=1

for ss in state_store.values():
    for s, stat in ss.items():
        if s<HORIZON:
            _, probs, values = get_acts_probs_values(stat)
            entropies, peaks = get_I(probs)
            jsds = get_jsd(stat)

            acting_agents = copy.deepcopy(list(AGENT_NAMES.values()))
            # Sample a random order of acting agents (sigma)
            random.shuffle(acting_agents)

            for acting in acting_agents:
                state_seq = {}
                for affected in AGENT_NAMES.values():
                    state_seq[affected] = seq_agent_obs_indiv(affected, acting, ss, s)

                _, probs_new, values_new = get_acts_probs_values(state_seq)
                jsds_new = get_jsd(state_seq)
                entropies_new, peaks_new = get_I(probs_new)

                if WITH_ABS:
                    metrics_abs["abs_v"][s][AGENT_KEYS[acting]].append(values[AGENT_KEYS[acting]])
                    metrics_abs["abs_p"][s][AGENT_KEYS[acting]].append(peaks[AGENT_KEYS[acting]])
                    metrics_abs["abs_j-team"][s][AGENT_KEYS[acting]].append(jsds[AGENT_KEYS[acting]][0])
                    metrics_abs["abs_j-opp"][s][AGENT_KEYS[acting]].append(jsds[AGENT_KEYS[acting]][1])

                for affected in AGENT_NAMES.values():
                        # history data only on opponents
                        if acting.startswith("adversary") and affected.startswith("agent") or acting.startswith("agent") and affected.startswith("adversary"):
                            metrics["diff_v"][s][AGENT_KEYS[acting]].append(values_new[AGENT_KEYS[affected]]-values[AGENT_KEYS[affected]])
                            metrics["diff_p"][s][AGENT_KEYS[acting]].append(peaks_new[AGENT_KEYS[affected]]-peaks[AGENT_KEYS[affected]])
                            metrics["diff_j-team"][s][AGENT_KEYS[acting]].append(jsds_new[AGENT_KEYS[affected]][0]-jsds[AGENT_KEYS[affected]][0])
                            metrics["diff_j-opp"][s][AGENT_KEYS[acting]].append(jsds_new[AGENT_KEYS[affected]][1]-jsds[AGENT_KEYS[affected]][1])
                        
                        metrics_indiv["diff_v"][s][AGENT_KEYS[acting]][AGENT_KEYS[affected]].append(values_new[AGENT_KEYS[affected]]-values[AGENT_KEYS[affected]])
                        metrics_indiv["diff_p"][s][AGENT_KEYS[acting]][AGENT_KEYS[affected]].append(peaks_new[AGENT_KEYS[affected]]-peaks[AGENT_KEYS[affected]])
                        metrics_indiv["diff_j-team"][s][AGENT_KEYS[acting]][AGENT_KEYS[affected]].append(jsds_new[AGENT_KEYS[affected]][0]-jsds[AGENT_KEYS[affected]][0])
                        metrics_indiv["diff_j-opp"][s][AGENT_KEYS[acting]][AGENT_KEYS[affected]].append(jsds_new[AGENT_KEYS[affected]][1]-jsds[AGENT_KEYS[affected]][1])

# Compute average values for each metric across all timesteps
avg_metrics = {}
avg_metrics_indiv = {}
for k,v in metrics.items():
    met = k.split("_")[1]
    avg_metrics[f"avg_{met}"] = avg_data(v)
for k,v in metrics_indiv.items():
    met = k.split("_")[1]
    avg_metrics_indiv[f"avg_{met}"] = avg_data(v, True)

# store history plots
write_data_history(list(steps), name="diff", **avg_metrics)
plot_history(list(steps), name="diff", **avg_metrics)
#plot_icv(**avg_metrics)

# store ICV data of all agents
write_data_icv(**avg_metrics_indiv)
plot_icv_indiv(**avg_metrics_indiv)


if WITH_ABS:
    avg_metrics_abs = {}
    for k,v in metrics_abs.items():
        met = k.split("_")[1]
        avg_metrics_abs[f"avg_{met}"] = avg_data(v)
        
    write_data_history(list(steps), name="abs", **avg_metrics_abs)
    plot_history(list(steps), name="abs", **avg_metrics_abs)

print("finish")