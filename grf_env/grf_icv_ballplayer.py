import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from g_env import GEnv
from tizero.tizero_policy import OpenRLAgent
from typing import Dict, Any, List, Tuple
from scipy.spatial.distance import jensenshannon

"""
ICV Computation for trio of ball player or striker in GFootball environment.

EFFECTS MEASUREMENT HERE:
if BALL_PLAYING = True:
    opp-> ball player & remaining -> ball player
else:
    opp-> striker & remaining -> striker

--> REPRODUCE HISTORY RESULTS: 
        ITERS = 1, BALL_PLAYING = True, MANUAL_SEED = 175, ICV_INTERVAL = 2, ALL_STATES = True
        With ITERS=1, the scenario 11_vs_11_icv_t500 will be loaded having game configuration: scenario duration = 500, difficulty = 1.0

--> Results stored under icv_store_ball folder
"""

#coalition partitions
P_S = 2 #striker
P_C = [0,1,3,4,5,6,7,8,9,10] #coalition
P_O = [0,1,2,3,4,5,6,7,8,9,10] #opponents

ITERS = 50
ICV_INTERVAL = 2 if ITERS == 1 else 5
MANUAL_SEED = 175
ALL_STATES = True #if True, use all states to compute ICV, irrespective if ball is played by player, otherwise only use states where ball is played by player
BALL_PLAYING = True # if True, ICV computed on ball playing player, else on striker

DIR = os.path.dirname(__file__)
store_path = os.path.join(DIR, "icv_store_ball")
if not os.path.exists(store_path):
    os.makedirs(store_path)

def intermediate_states(
    state_t0: Dict[str, Any],
    state_t1: Dict[str, Any],
    p_c:    List[int],
    p_o:   List[int],
) -> Tuple[Dict[str,Any], Dict[str,Any], Dict[str,Any], Dict[str,Any]]:
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

    # Condition to decide when to apply general info
    cond = state_t0.get("ball_owned_team") in (1, -1)

    # s_0 is just the original
    s_0 = copy.deepcopy(state_t0)

    # Helper to update a snapshot from base by copying subset of keys/indices
    def _update(snapshot, base, keys, indices):
        for k in keys:
            arr0 = snapshot[k]
            arr1 = base[k]
            for i in indices:
                arr0[i] = copy.deepcopy(arr1[i])
        return snapshot

    # Build s_1
    s_1 = copy.deepcopy(state_t0)
    s_1 = _update(s_1, state_t1, RIGHT_KEYS, p_o)
    if cond:
        for g in GENERAL:
            s_1[g] = copy.deepcopy(state_t1[g])

    # Build s_2
    s_2 = copy.deepcopy(s_1)
    s_2 = _update(s_2, state_t1, LEFT_KEYS, p_c)
    if not cond:
        for g in GENERAL:
            s_2[g] = copy.deepcopy(state_t1[g])

    return s_0, s_1, s_2

def get_env(seed:int=2):
    """
    Get GFootball environment with specified seed.
    """
    scneario = "GFootball/11_vs_11_icv_t500" if ITERS == 1 else "GFootball/11_vs_11_icv"
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

def get_jsd(agent,state1, state2):
    """
    Compute JSD for given states.
    """
    _, probs1, _ = agent.get_action_probs(state1, all_acts=False, rnn=False)
    _, probs2, _ = agent.get_action_probs(state2, all_acts=False, rnn=False)
    probs1 = probs1[0].tolist()
    probs2 = probs2[0].tolist()
    jsd = jensenshannon(probs1, probs2, 2)
    if math.isnan(jsd): jsd = 0 #sometimes happens that jsd is nan.
    return jsd

def plot_history(steps, opp_metrics, coal_metrics):
    """
    Plot ICV history for opponent and coalition effects.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(steps, opp_metrics["val"], label='v_o', color='black')
    ax[0].plot(steps, opp_metrics["peak"], label='p_o', color='green')
    ax[0].plot(steps, opp_metrics["jsd"], label='j_o', color='orange')
    ax[0].set_title('Opponent Effect')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Value Difference')
    ax[0].legend()
    
    ax[1].plot(steps, coal_metrics["val"], label='v_c', color='black')
    ax[1].plot(steps, coal_metrics["peak"], label='p_c', color='green')
    ax[1].plot(steps, coal_metrics["jsd"], label='j_c', color='orange')
    ax[1].set_title('Coalition Effect')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Value Difference')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, "icv_store_ball", f"ICV_ball_hist.png"))
    # plt.show()

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
    plt.savefig(os.path.join(DIR, "icv_store_ball", f"ICV_ball_plots.png"))
    # plt.show()

def write_data_history(steps, opp_metrics, coal_metrics):
    """
    Store ICV history data to file.
    """
    filename= os.path.join(DIR, "icv_store_ball", "ICV_ball_history.dat")
    with open(filename, "w") as f:
        f.write("x t v_o p_o j_o v_c p_c j_c\n")

        # loop through all time steps
        for i in range(len(steps)):
            line = f"{steps[i]} {steps[i]*0.03} {opp_metrics['val'][i]} {opp_metrics['peak'][i]} {opp_metrics['jsd'][i]} {coal_metrics['val'][i]} {coal_metrics['peak'][i]} {coal_metrics['jsd'][i]}\n"
            f.write(line)

full_opp_metrics = {"val":[], "peak":[], "jsd":[]}
full_coal_metrics = {"val":[], "peak":[], "jsd":[]}

print("Start game and ICV computation for ball player or striker...")

for its in range(ITERS):
    seed = its*50 + np.random.randint(1, 50) if ITERS > 1 else MANUAL_SEED
    env = get_env(seed=seed)
    agent = OpenRLAgent()
    obs, _ = env.reset()
    done = False
    state_store = {}
    steps = set()
    s = 0

    while not done:
        acts = np.zeros(len(obs), dtype=np.int32)
        
        # ball playing player only, else striker
        ball_player = P_S
        if BALL_PLAYING:
            if obs[1].get("ball_owned_team") == 0 and obs[1].get("ball_owned_player") > 0:
                ball_player = obs[1].get("ball_owned_player")
        state_store[s] = obs[ball_player]

        for k, v in enumerate(obs):
            act, probs, _ = agent.get_action_probs(v, all_acts=False) #True w/o action masking
            acts[k] = np.argmax(act[0])

        obs, _, dones, _ = env.step(acts)
        done = all(dones)
        if not done:
            steps.add(s)
            s+=1

    coal_metrics = {"val":[], "peak":[], "jsd":[]}
    opp_metrics = {"val":[], "peak":[], "jsd":[]}
    ssteps = set()

    # Post-processing of states
    for ks in steps:
        if ks % ICV_INTERVAL == 0:
            #if state_store[ks]["ball_owned_team"] >=0 and state_store[ks+1]["ball_owned_team"] >=0:
            if ALL_STATES or state_store[ks]["ball_owned_team"] >=0:
                if BALL_PLAYING:
                    ball_player = state_store[ks]["ball_owned_player"]
                    p_c = [i for i in range(11) if i != ball_player]

                # intermediate states according to coalition partitions
                s0, s1, s2 = intermediate_states(
                    state_t0=state_store[ks],
                    state_t1=state_store[ks+1],
                    p_c=p_c if BALL_PLAYING else P_C,
                    p_o=P_O
                )

                opp_metrics["val"].append(agent.get_value(s1)-agent.get_value(s0))
                opp_metrics["peak"].append(get_I(agent,s1)[1] - get_I(agent,s0)[1])
                opp_metrics["jsd"].append(get_jsd(agent,s0, s1))

                coal_metrics["val"].append(agent.get_value(s2)-agent.get_value(s1))
                coal_metrics["peak"].append(get_I(agent,s2)[1] - get_I(agent,s1)[1])
                coal_metrics["jsd"].append(get_jsd(agent,s1, s2))

            # else:
            #     opp_metrics["val"].append(0)
            #     opp_metrics["peak"].append(0)
            #     opp_metrics["jsd"].append(0)

            #     coal_metrics["val"].append(0)
            #     coal_metrics["peak"].append(0)
            #     coal_metrics["jsd"].append(0)
                
                ssteps.add(ks)

    if ITERS == 1:
        plot_history(sorted(list(ssteps)), opp_metrics, coal_metrics)
        write_data_history(sorted(list(ssteps)), opp_metrics, coal_metrics)

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
with open(os.path.join(DIR, "icv_store_ball", "vals.txt"), "w") as f:
    f.write(f"opp metrics: {full_opp_metrics}\n")
    f.write(f"coal metrics: {full_coal_metrics}\n")

print(f"finish game")