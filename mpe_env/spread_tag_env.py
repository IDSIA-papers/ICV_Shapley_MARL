from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo.mpe import simple_spread_v3, simple_tag_v3


class SpreadEnv(MultiAgentEnv):
    def __init__(self, config):
        self._skip_env_checking = True
        self.config = config
        self.n_agents = config["n_agents"]
        self.local_ratio = config["local_ratio"]
        self.horizon = config["max_cylces"]
        self.continuous_actions = config["continuous_actions"]
        self.render_mode = config["render_mode"]
        self.seed = config["seed"]
        self.rew_scale = config["rew_scale"]

        self.env = simple_spread_v3.parallel_env(N=self.n_agents, local_ratio=self.local_ratio, 
                                        max_cycles=self.horizon, continuous_actions=self.continuous_actions, 
                                        render_mode=self.render_mode)
        self.env.reset(self.seed)

        self.action_space=spaces.Dict()
        self.observation_space=spaces.Dict()

        #env.agents contains list of agent names
        for a in self.env.agents:
            self.action_space[a] = self.env.action_space(a)
            self.observation_space[a] = self.env.observation_space(a)

        self._agent_ids = set(self.env.agents) # {'agent_0', 'agent_1', etc.}

        super().__init__()


    def reset(self, *, seed=None, options=None):
        return self.env.reset(self.seed)

    def step(self, actions):
        obs, rews, term, trunc, info = self.env.step(actions)
        term["__all__"] = all(term.values())
        trunc["__all__"] = all(trunc.values())
        rewards = {k:rews[k] for k in list(actions.keys())}
        for ag in self._agent_ids:
            if ag not in list(obs.keys()): obs[ag] = np.zeros(self.observation_space[ag].shape, dtype=np.float32)
        assert sorted(list(self._agent_ids)) == sorted(list(obs.keys())), f"obs keys {list(obs.keys())} is not {list(self._agent_ids)}"
        return obs, rewards, term, trunc, info
        
    def render(self):
        self.env.render()

class TagEnv(MultiAgentEnv):
    """
    Ray Wrapper for the simple_tag_v3 environment from PettingZoo.

    In original simple_tag.py, ADJUST observation method as follows:

    def observation(self, agent, world):
        custom_order = []
        custom_vals = []
        if not agent.adversary:
            custom_order.extend(self.adversaries(world))
        else:
            custom_order.extend(self.good_agents(world))

        for other in custom_order:
            custom_vals.append(other.state.p_pos - agent.state.p_pos)
            custom_vals.append(other.state.p_vel)

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + custom_vals
        )
    """
    def __init__(self, config):
        self._skip_env_checking = False
        self.config = config
        self.num_good = config["num_good"]
        self.num_adv = config["num_adv"]
        self.num_obst = config["num_obst"]
        self.horizon = config["max_cylces"]
        self.continuous_actions = config["continuous_actions"]
        self.render_mode = config["render_mode"]
        self.seed = config["seed"]

        self.env = simple_tag_v3.parallel_env(num_good=self.num_good, num_adversaries=self.num_adv, num_obstacles=self.num_obst,
                                        max_cycles=self.horizon, continuous_actions=self.continuous_actions, 
                                        render_mode=self.render_mode)
        self.env.reset(self.seed)

        self.action_space=spaces.Dict()
        self.observation_space=spaces.Dict()

        #env.agents contains list of agent names
        for a in self.env.agents:
            self.action_space[a] = self.env.action_space(a)
            self.observation_space[a] = self.env.observation_space(a)

        self._agent_ids = set(self.env.agents) # {'agent_0', 'agent_1', etc.}
        self.num_agents = self.num_adv+self.num_good
        super().__init__()


    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed)

    def step(self, actions):
        obs, rews, term, trunc, info = self.env.step(actions)
        term["__all__"] = all(term.values())
        trunc["__all__"] = all(trunc.values())
        rewards = {k:rews[k] for k in list(actions.keys())}
        for ag in self._agent_ids:
            if ag not in list(obs.keys()): obs[ag] = np.zeros(self.observation_space[ag].shape, dtype=np.float32)
        assert sorted(list(self._agent_ids)) == sorted(list(obs.keys())), f"obs keys {list(obs.keys())} is not {list(self._agent_ids)}"
        return obs, rewards, term, trunc, info
        
    def render(self):
        self.env.render()