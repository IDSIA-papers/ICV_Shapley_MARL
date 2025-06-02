from lbf_env import LBFEnv

"""
Test script for the LBF environment, with manual input for actions.
"""

#ACTIONS:
"""
NONE = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
LOAD = 5
"""

config = {
    "level":3,
    "players":3,
    "size":9,
    "max_num_food":3,
    "coop": True,
    "rew_pot":True,
    "render":None
}

env = LBFEnv(config)
obs,_ = env.reset()
done = False

while not done:
    env.render()
    a = 0
    while not a:
        a = input("input: ")
        a = {k:int(n) for k,n in enumerate(a)}
        # a = tuple([int(act) for act in a])

    o,r,d,_,_ = env.step(a)
    done = d["__all__"]
