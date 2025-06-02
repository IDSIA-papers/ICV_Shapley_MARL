import matplotlib.pyplot as plt
import os
import torch
import math
import random
from scipy.spatial.distance import jensenshannon
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.algorithm import Algorithm
from spread_tag_env import TagEnv
from model.mpe_net import TagNetAgentSymmIndep

ray.shutdown()
ray.init(ignore_reinit_error=True, local_mode=False)

"""
File for plotting the HISTORY of values of the MPE Tag environment. No ICV, yet.
You can manually specify which values to plot by changing the `EXCLUDE_PLOT` variable.
"""

MODE = 0 #MODE=0: absolute values, 1= differences between two consecutive steps
ITERS = 1

ACTS = {0: "0:NOP", 1:"1:LEFT", 2:"2:RIGHT", 3:"3:DOWN", 4:"4:UP"}
AGENT_NAMES = {0:'adversary_0', 1:'adversary_1', 2:'agent_0', 3: 'agent_1'}
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
EXCLUDE_PLOT = ["h", "jc"] #ignore these in plots

store_path = os.path.join(DIR, "results_store")
if not os.path.exists(store_path):
    os.makedirs(store_path)

def get_algo():
    ModelCatalog.register_custom_model("tag_model_agent", TagNetAgentSymmIndep)
    rest_path = os.path.join(DIR, "model", f"tag_e0_kl02", "checkpoint")
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
    for i in AGENTS:
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
    for i in AGENTS:
        acts[AGENT_NAMES[i]],_,info = policies[AGENT_NAMES[i]].compute_single_action(obs=state[AGENT_NAMES[i]], explore=False)
        probs[i] = torch.softmax(torch.tensor(info["action_dist_inputs"]), dim=0).tolist()
        values[i] = info["vf_preds"]
    return acts, probs, values

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
                ax.set_title(AGENT_NAMES[k])
                kol+=1
    
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "results_store", f"INFERENCE_history_{name}.png"))
    plt.close()

def write_data(steps, name="", **kwargs):
    """
    Store data as following example, where x = step, and val0, val1, val2 are the values for agents 0, 1, and 2 respectively.
    x v0 v1 v2 h0 h1 h2 p0 p1 p2 c0 c1 c2 js0 js1 js2 jo0 jo1 jo2
    0 0.5 0.6 0.7 0.1 0.2 0.3 0.4 0.5 0.6 1.0 1.1 1.2 0.9 1.0 1.1
    ... 
    """
    filename = os.path.join(DIR, "results_store", f"mpe_data_{name}.dat")
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
    avg_data = {k:[] for k in AGENTS}
    for v_s in data.values(): #timestep values
        for k, v_k in v_s.items():
            if len(v_k) > 0:
                avg_data[k].append(sum(v_k) / len(v_k))
            else:
                avg_data[k].append(0.0)
    return avg_data

algo = get_algo()
policies = get_policies(algo)
os.system('clear') if os.name == 'posix' else os.system('cls')

#metrics store either abs values or abs differences during raw inference (no contributions of other players)
metrics = {
    "v": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "h": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "p": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "jc": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
    "jd": { s:{i:[] for i in AGENTS} for s in range(HORIZON)},
}

steps = set()

for it in range(ITERS):
    print(f"iteration {it}")
    state, _ = env.reset(seed=random.randint(0, 100))
    done = False
    s = 0

    while not done:
        acts, probs,values = get_acts_probs_values(state)
        entropies, peaks = get_I(probs)
        jsds = get_jsd(state)

        #absolute values
        if MODE == 0:
            for ag_id in AGENTS:
                metrics["v"][s][ag_id].append(values[ag_id])
                metrics["h"][s][ag_id].append(entropies[ag_id])
                metrics["p"][s][ag_id].append(peaks[ag_id])
                metrics["jc"][s][ag_id].append(jsds[ag_id][0])
                metrics["jd"][s][ag_id].append(jsds[ag_id][1])
            state,_,term, trunc, _ = env.step(acts)

        #differences
        elif MODE == 1:
            state,_,term,trunc, _ = env.step(acts)

            acts_new, probs_new, values_new = get_acts_probs_values(state)
            entropies_new, peaks_new = get_I(probs_new)
            jsds_new = get_jsd(state)

            for ag_id in AGENTS:
                metrics["v"][s][ag_id].append(values_new[ag_id]-values[ag_id])
                metrics["h"][s][ag_id].append(entropies_new[ag_id]-entropies[ag_id])
                metrics["p"][s][ag_id].append(peaks_new[ag_id]-peaks[ag_id])
                metrics["jc"][s][ag_id].append(jsds_new[ag_id][0]-jsds[ag_id][0])
                metrics["jd"][s][ag_id].append(jsds_new[ag_id][1]-jsds[ag_id][1])

        done = term["__all__"] or trunc["__all__"]
        steps.add(s)
        s+=1

avg_metrics = {}
for k, v in metrics.items():
    avg_metrics[f"{k}"] = avg_data(v)

plot_history(steps=list(steps), name="abs" if MODE == 0 else "diff", **avg_metrics)
write_data(list(steps), name="abs" if MODE == 0 else "diff", **avg_metrics)

print("finish")