import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from g_env import GEnv
from tizero.tizero_policy import OpenRLAgent
from scipy.spatial.distance import jensenshannon

"""
ICV Computation for trio of players (RM, CF, LM) in GFootball environment.

EFFECTS MEASUREMENT HERE:
opp-> RM, CF, LM & remaining -> RM, CF, LM

--> Results stored under icv_store_trio folder
"""

ITERS = 50
INTERVAL = 5
ALL_STATES = True #if True, use all states to compute ICV, irrespective if ball is played by player, otherwise only use states where ball is played by player

P_sub = [1, 2, 10] #role indices of positions RM, CF, LM
P_sub_idx = {1: "RM", 2: "CF", 10: "LM"}

DIR = os.path.dirname(__file__)
store_path = os.path.join(DIR, "icv_store_trio")
if not os.path.exists(store_path):
    os.makedirs(store_path)

def intermediate_state(st0, st1, right, ball_players=None):
    """
    Compute intermediate state between two states st0 and st1.
    """

    GENERAL = [
        "ball", "ball_direction", "ball_rotation",
        "ball_owned_player", "ball_owned_team",
        "steps_left", "game_mode", "score"
    ]
    RIGHT_KEYS = [
        "right_team", "right_team_direction",
        "right_team_tired_factor", "right_team_yellow_card",
        "right_team_active", "right_team_roles"
    ]
    LEFT_KEYS = [
        "left_team", "left_team_direction",
        "left_team_tired_factor", "left_team_yellow_card",
        "left_team_active", "left_team_roles"
    ]
    if right:
        keys = RIGHT_KEYS
        indices = [i for i in range(11)]
        cond = st0.get("ball_owned_team") == 1 and st0.get("ball_owned_player") > 0
    else:
        keys = LEFT_KEYS
        indices = [i for i in range(11) if i not in ball_players]
        cond = st0.get("ball_owned_team") == 0 and st0.get("ball_owned_player") > 0

    s_1 = copy.deepcopy(st0)
    for k in keys:
        for i in indices:
            s_1[k][i] = copy.deepcopy(st1[k][i])
    if cond:
        for g in GENERAL:
            s_1[g] = copy.deepcopy(st1[g])

    return s_1

def get_env(seed:int=3):
    """
    Get GFootball environment with specified seed.
    """
    scneario = "GFootball/11_vs_11_icv"
    scenario_name = scneario.split("/")[-1]
    config= {"left_agent_num":11, "right_agent_num":0}
    env = GEnv(cfg=config, 
               scenario_name=scenario_name, 
               log_dir=os.path.join(DIR, "dumps"),
               seed=seed,
               dumps=False,
               render=False)
    return env

def get_I(agent,state):
    """
    Compute entropy and peakedness of the action probabilities for a given state.
    """
    _, probs, _ = agent.get_action_probs(state, all_acts=False, rnn=False)
    probs = probs[0].tolist()
    maxi = math.log2(len(probs))
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    p = maxi - h
    return h/maxi, p/maxi

def plot_bars(metrics):
    """
    Plot ICV bars for opponent and coalition effects.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].bar(metrics["opp"].keys(), metrics["opp"].values(), color='black')
    ax[0].set_title('Opponent Effect')
    ax[0].set_ylabel('Difference')
    
    ax[1].bar(metrics["coal"].keys(), metrics["coal"].values(), color='green')
    ax[1].set_title('Coalition Effect')
    ax[1].set_ylabel('Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "icv_store_trio", f"ICV_trio_plots.png"))
    # plt.show()

full_opp_metrics = {"val":[], "peak_RM":[],"peak_CF":[],"peak_LM":[]}
full_coal_metrics = {"val":[], "peak_RM":[], "peak_CF":[], "peak_LM":[]}

print("Start game and ICV computation for forwarding trio...")

for its in range(ITERS):
    seed = its*50 + np.random.randint(1, 50)
    env = get_env(seed=seed)
    agent = OpenRLAgent()
    obs, _ = env.reset()
    done = False
    state_store_all = {}
    steps = set()
    s = 0

    while not done:
        acts = np.zeros(len(obs), dtype=np.int32)

        # store states of forwarding trio
        state_store_all[s] = [obs[k] for k in P_sub]

        for k, v in enumerate(obs):
            act, probs, entropy = agent.get_action_probs(v, all_acts=False) #True w/o action masking
            acts[k] = np.argmax(act[0])
            
        obs, _, dones, _ = env.step(acts)
        done = all(dones)
        if not done:
            steps.add(s)
            s+=1

    opp_metrics = {"val":[], "peak_RM":[], "peak_CF":[], "peak_LM":[]}
    coal_metrics = {"val":[], "peak_RM":[], "peak_CF":[], "peak_LM":[]}
    ssteps = set()

    # Post-processing of states
    for ks in steps:
        if ks%INTERVAL == 0:
            if ALL_STATES or state_store_all[ks][1]["ball_owned_team"] >= 0:

                opp_inter_state = None
                coal_inter_state = None

                for k, (ag_state_t0, ag_state_t1) in enumerate(zip(state_store_all[ks], state_store_all[ks+1])): #list of agent states in P_sub (RM, CF, LM)
                    if ag_state_t0["controlled_player_index"] > 0:
                        opp_state = intermediate_state(ag_state_t0, ag_state_t1, right=True)
                        ag_i = ag_state_t0["controlled_player_index"]
                        opp_metrics[f"peak_{P_sub_idx[ag_i]}"].append(get_I(agent,ag_state_t0)[1] - get_I(agent,opp_state)[1])
                        
                        coal_state = intermediate_state(opp_state, ag_state_t1, right=False, ball_players=[ag_i])
                        coal_metrics[f"peak_{P_sub_idx[ag_i]}"].append(get_I(agent,coal_state)[1] - get_I(agent,opp_state)[1])
                        
                        if k == len(P_sub) - 1: #last agent in P_sub
                            opp_inter_state = copy.deepcopy(opp_state)
                            coal_inter_state = copy.deepcopy(coal_state)

                opp_metrics["val"].append(agent.get_value(opp_inter_state) - agent.get_value(state_store_all[ks][-1]))
                coal_metrics["val"].append(agent.get_value(coal_inter_state) - agent.get_value(opp_inter_state))
            
            # else:
            #     for k in P_sub:
            #         opp_metrics[f"peak_{P_sub_idx[k]}"].append(0.0)
            #         coal_metrics[f"peak_{P_sub_idx[k]}"].append(0.0)
            #     opp_metrics["val"].append(0.0)
            #     coal_metrics["val"].append(0.0)
                
                ssteps.add(ks)

    # Store metrics for this iteration
    for k in opp_metrics.keys():
        full_opp_metrics[k].extend(opp_metrics[k])
    for k in coal_metrics.keys():
        full_coal_metrics[k].extend(coal_metrics[k])

    print("Iteration", its, "done")

# Average over all iterations
for k in full_opp_metrics.keys():
    full_opp_metrics[k] = sum(full_opp_metrics[k]) / len(full_opp_metrics[k])
for k in full_coal_metrics.keys():
    full_coal_metrics[k] = sum(full_coal_metrics[k]) / len(full_coal_metrics[k])

plot_bars({"opp": full_opp_metrics,"coal": full_coal_metrics})

# Store full metrics to file
with open(os.path.join(DIR, "icv_store_trio", "vals.txt"), "w") as f:
    f.write(f"opp metrics: {full_opp_metrics}\n")
    f.write(f"coal metrics: {full_coal_metrics}\n")

print(f"finish game")