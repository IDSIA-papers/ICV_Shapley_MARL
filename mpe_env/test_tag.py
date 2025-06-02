import os
import ray
import time
from spread_tag_env import TagEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from model.mpe_net import TagNetAgentSymmIndep

"""
This file tests the Tag environment using a pre-trained Ray RLlib model. 
Attention: check the description in environment 'spread_tag_env.py' for the correct observation modification!
"""

N_GOOD= 2
N_ADV = 2
N_OB = 0
R_PATH = os.path.join(os.path.dirname(__file__), 'model', 'tag_e001', 'checkpoint')

ray.init()
env_config_p={
    "num_good": N_GOOD,
    "num_adv": N_ADV,
    "num_obst": N_OB,
    "max_cylces": 50,
    "continuous_actions": False,
    "render_mode": None, #"human",
    "seed": None
}

ModelCatalog.register_custom_model("tag_model_agent", TagNetAgentSymmIndep)

algo = Algorithm.from_checkpoint(R_PATH)
env = TagEnv(env_config_p)

for it in range(5):
    state, _ = env.reset()
    reward =0
    done = False
    while not done:
        env.render()
        actions = {}
        for ag_id, stat in state.items():
            actions[ag_id] = algo.compute_single_action(observation=stat, policy_id=f"{ag_id}", explore=False)

        state, rew, term, trunc, _ = env.step(actions)

        done = term["__all__"] or trunc["__all__"]
        for r in rew.values():
            reward += r

    print(f"Iteration {it}, REWARDS {reward:.2f}")
