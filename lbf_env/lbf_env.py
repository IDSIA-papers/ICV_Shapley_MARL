from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from foraging.env import ForagingEnv

class LBFEnv(MultiAgentEnv):
    """LBF Environment wrapper for Ray RLlib."""

    def __init__(self, config):
        #self._skip_env_checking = True
        self.config = config

        self.env = ForagingEnv(
            game_level=config.get("level",None),
            players=config.get("players", 2),
            min_player_level=1,
            max_player_level=2,
            min_food_level=1,
            max_food_level=None,
            field_size=(config.get("size",5), config.get("size",5)),
            max_num_food=config.get("max_num_food", 5),
            sight=config.get("sight", config.get("size", 5)),
            max_episode_steps=config.get("horizon", 50),
            force_coop=config.get("coop", True),
            load_punish=config.get("load_punish", False),
            reward_potential=config.get("rew_pot", False),
            normalize_reward=config.get("norm_rew", True),
            render_mode=config.get("render", None)
        )

        self.num_agents = config.get("players", 2)
        self.action_space=spaces.Dict()
        self.observation_space=spaces.Dict()

        for a in range(self.num_agents):
            self.action_space[a] = self.env.action_space[a]
            self.observation_space[a] = self.env.observation_space[a]

        self._agent_ids = set(range(self.num_agents))

        super().__init__()

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action_dict, count=True):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        obs, rewards, d, t, i = self.env.step(actions=tuple(actions), count=count)
        return obs, rewards, {"__all__": d}, {"__all__": t}, i
       
    def render(self):
        self.env.render()

