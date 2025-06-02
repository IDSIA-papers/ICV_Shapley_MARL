import copy
import gymnasium as gym
from typing import Dict, Optional
import numpy as np
import football_env

class GEnv(gym.Env):
    """
    GFootball Environment Wrapper.
    """
    def __init__(
        self,
        scenario_name: str,
        cfg: Dict = {},
        log_dir=None,
        seed: int = 0,
        render=False,
        write_video=False,
        dumps=False,
    ):

        assert seed is not None, "you must set a seed when setup the environment"

        self.left_agent_num = cfg.pop("left_agent_num", 11)
        self.right_agent_num = cfg.pop("right_agent_num", 11)

        video_quality_level = 0

        representation = "raw"  # simple115, simple115v2,extracted

        self.env = football_env.create_environment(
            env_name=scenario_name, #11_vs_11_icv
            stacked=False,
            logdir=log_dir, #dumps
            representation=representation,
            rewards="scoring",
            write_goal_dumps=False,
            write_full_episode_dumps=dumps,
            render=render,
            write_video=write_video,
            dump_frequency=1 if dumps else 0,
            extra_players= None, #["bot:right_players=1"],
            number_of_left_players_agent_controls=self.left_agent_num,
            number_of_right_players_agent_controls=self.right_agent_num,
            other_config_options={
                "action_set": "v2", #"full" for bot (extra player)
                "video_quality_level": video_quality_level,
                "game_engine_random_seed": seed,
                "physics_steps_per_frame":10 # original = 10
            },
        )

        self.cur_step = 0

    def reset(self):
        raw_obs = self.env.reset()
        self.cur_step = 0
        return self.add_control_index(raw_obs), [
            {} for i in range(self.left_agent_num + self.right_agent_num)
        ]

    def add_control_index(self, raw_obs):
        """
        For 11 players of TiZero, both are the same, otherwise there is a mismatch.
        """
        for i, o in enumerate(raw_obs):
            if "controlled_player_index" not in o:
                o["controlled_player_index"] = i % 11
        # for o in raw_obs:
        #     if "controlled_player_index" not in o:
        #         o["controlled_player_index"] = o["active"] % self.left_agent_num
        return raw_obs

    def step(self, actions):
        self.cur_step += 1
        actions = np.array(actions)

        if len(actions.shape) != 1:
            actions = np.argmax(actions, axis=-1)

        raw_o, r, d, info = self.env.step(actions.astype("int32"))
        info["enemy_designated"] = raw_o[-1]["designated"]
        dones = []
        infos = []

        for i in range(self.left_agent_num + self.right_agent_num):
            dones.append(d)
            reward = r[i] if self.left_agent_num > 1 else r
            info["goal"] = reward
            infos.append(copy.deepcopy(info))

        return self.add_control_index(raw_o), r, dones, infos

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
