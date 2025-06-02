"""
Test script for GFootball environment using TiZero policy.
Official TiZero implementation: https://github.com/OpenRL-Lab/TiZero

Within scenario file, when deterministic = True, setting seed has no effect, as the environment will automatically set game_engine_random_seed = 42.
When deterministic = False, if game_engine_random_seed is not set, the environment will initialize with a random seed.
If we want to set the seed manually, we need to set deterministic to False and then specify game_engine_random_seed!
"""

import os
import numpy as np
from g_env import GEnv
from tizero.tizero_policy import get_action_probs, get_value

## ACTION SET
actions_dict = {
    0: "NO_OP",
    1: "LEFT",
    2: "TOP_LEFT",
    3: "TOP",
    4: "TOP_RIGHT",
    5: "RIGHT",
    6: "BOTTOM_RIGHT",
    7: "BOTTOM",
    8: "BOTTOM_LEFT",
    9: "LONG_PASS",
    10: "HIGH_PASS",
    11: "SHORT_PASS",
    12: "SHOT",
    13: "SPRINT",
    14: "RELEASE_DIRECTION",
    15: "RELEASE_SPRINT",
    16: "SLIDE",
    17: "DRIBBLE",
    18: "RELEASE_DRIBBLE",
    19: "NONE"
}


scneario = "GFootball/11_vs_11_icv"
scenario_name = scneario.split("/")[-1]
config= {"left_agent_num":11, "right_agent_num":0}

env = GEnv(cfg=config, 
           scenario_name=scenario_name, 
           log_dir=os.path.join(os.path.dirname(__file__), "dumps"),
           seed=175, #if deterministic scenario, seed will be 42
           dumps=False,
           render=False)

obs, info = env.reset()
done = False
acts = np.zeros(len(obs), dtype=np.int32)
i = 0

while not done:
    for k, v in enumerate(obs):
        act, act_probs, _ = get_action_probs(v, all_acts=False) #True w/o action masking
        acts[k] = np.argmax(act[0])
        # if k> 0:
        #     val = get_value(v, rnn=True)
        #     print("val ", val)

    obs, reward, dones, infos = env.step(acts)
    done = all(dones)
    if sum(reward)>0:
        print("goal scored at step ", i)
    elif sum(reward)<0:
        print("goal got at step ", i)
    i+=1
    
print("finish game!")
