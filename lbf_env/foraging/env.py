from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
import logging
from typing import Iterable

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(gym.Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 5,
    }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        game_level,
        players,
        min_player_level,
        max_player_level,
        min_food_level,
        max_food_level,
        field_size,
        max_num_food,
        sight,
        max_episode_steps,
        force_coop,
        load_punish=False,
        reward_potential=False,
        normalize_reward=True,
        grid_observation=False,
        observe_agent_levels=True,
        penalty=0.0,
        render_mode=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32) #used for game dynamics
        self.grid = np.zeros(field_size, np.int32) #used for grid obs

        self.penalty = penalty
        self.level = game_level

        if isinstance(min_food_level, Iterable):
            assert (
                len(min_food_level) == max_num_food
            ), "min_food_level must be a scalar or a list of length max_num_food"
            self.min_food_level = np.array(min_food_level)
        else:
            self.min_food_level = np.array([min_food_level] * max_num_food)

        if max_food_level is None:
            self.max_food_level = None
        elif isinstance(max_food_level, Iterable):
            assert (
                len(max_food_level) == max_num_food
            ), "max_food_level must be a scalar or a list of length max_num_food"
            self.max_food_level = np.array(max_food_level)
        else:
            self.max_food_level = np.array([max_food_level] * max_num_food)

        if self.max_food_level is not None:
            # check if min_food_level is less than max_food_level
            for min_food_level, max_food_level in zip(
                self.min_food_level, self.max_food_level
            ):
                assert (
                    min_food_level <= max_food_level
                ), "min_food_level must be less than or equal to max_food_level for each food"

        self.max_num_food = max_num_food
        self._food_spawned = 0.0

        if isinstance(min_player_level, Iterable):
            assert (
                len(min_player_level) == players
            ), "min_player_level must be a scalar or a list of length players"
            self.min_player_level = np.array(min_player_level)
        else:
            self.min_player_level = np.array([min_player_level] * players)

        if isinstance(max_player_level, Iterable):
            assert (
                len(max_player_level) == players
            ), "max_player_level must be a scalar or a list of length players"
            self.max_player_level = np.array(max_player_level)
        else:
            self.max_player_level = np.array([max_player_level] * players)

        if self.max_player_level is not None:
            # check if min_player_level is less than max_player_level for each player
            for i, (min_player_level, max_player_level) in enumerate(
                zip(self.min_player_level, self.max_player_level)
            ):
                assert (
                    min_player_level <= max_player_level
                ), f"min_player_level must be less than or equal to max_player_level for each player but was {min_player_level} > {max_player_level} for player {i}"

        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self.reward_potential = reward_potential
        self.load_punish = load_punish
        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        self._observe_agent_levels = observe_agent_levels

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([gym.spaces.Box(low=0, high=20, shape=(86,),dtype=np.float32)] * len(self.players))
        )

        self.viewer = None

        self.num_agents = len(self.players)

        self.level = game_level
        self.food_pos = []
        self.player_pos = []
        self.prev_food_pos = None
        self.prev_player_pos = None
        self.prev_dist = None

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(
            players,
            min_player_level=1,
            max_player_level=2,
            min_food_level=1,
            max_food_level=None,
            field_size=None,
            max_num_food=None,
            sight=None,
            max_episode_steps=50,
            force_coop=False,
        )

        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_num_food, min_levels, max_levels):
        food_count = 0
        min_levels = max_levels if self.force_coop else min_levels

        # permute food levels
        food_permutation = self.np_random.permutation(max_num_food)
        min_levels = min_levels[food_permutation]
        max_levels = max_levels[food_permutation]

        for pos in self.food_pos:
            #for grid obs
            self.grid[pos[0], pos[1]] = 2

            self.field[pos[0], pos[1]] = (
                min_levels[food_count]
                if min_levels[food_count] == max_levels[food_count]
                else self.np_random.integers(
                    min_levels[food_count], max_levels[food_count] + 1
                )
            )
            food_count += 1

        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, min_player_levels, max_player_levels):
        # permute player levels
        player_permutation = self.np_random.permutation(len(self.players))
        min_player_levels = min_player_levels[player_permutation]
        max_player_levels = max_player_levels[player_permutation]

        for i, player, min_player_level, max_player_level in zip(
            range(self.num_agents) ,self.players, min_player_levels, max_player_levels
        ):
            player.setup(
                self.player_pos[i],
                self.np_random.integers(min_player_level, max_player_level + 1),
                self.field_size,
            )

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _dist_cell(self, first_loc, second_loc):
        return np.linalg.norm((np.array(first_loc) - np.array(second_loc)), 1)

    def _get_obs(self):
        obs = {}
        player_grid = np.zeros(self.field_size, np.int32)
        #target = self.fix_target()

        for player in self.players:
            player_grid[player.position[0], player.position[1]] = 1

        obs_grid = player_grid + self.grid
        grid_flat = obs_grid.reshape(-1)
        
        for i, player in enumerate(self.players):
            agent_loc = (player.position[0], player.position[1])
            dist = self._dist_cell(self.target_pos, agent_loc)
            obs[i] = np.concatenate((grid_flat, np.array(agent_loc), np.array([dist, self.target_pos[0], self.target_pos[1]])), axis=0, dtype=np.float32)
        return obs

    def fix_target(self):
        # indices = np.where(self.grid == 2)
        # result = list(zip(indices[0], indices[1]))
        # return result[0]

        # if init:
        #     self.target_pos=self.food_pos[np.random.choice(len(self.food_pos))]
        # else:
        #     possible = False
        #     p = None
        #     while not possible:
        #         p = self.food_pos[np.random.choice(len(self.food_pos))]
        #         possible = bool(self.field[p[0], p[1]])
        #     self.target_pos = p

        self.target_pos=self.food_pos[np.random.choice(len(self.food_pos))]

    def _gen_coord(self, min_val, max_val, m):
        coordinates = set()
        counter = 0
        while len(coordinates) < m:
            if counter >= 10000:
                #print(f"Warning: Max attempts (5000) reached in _gen_coord")
                break
            x, y = self.np_random.integers(min_val, max_val, endpoint=True), self.np_random.integers(min_val, max_val, endpoint=True)
            #if all(abs(x - cx) >= 1 and abs(y - cy) >= 1 for cx, cy in coordinates):
            if all( self._dist_cell((x_c, y_c), (x,y)) >=3 for x_c,y_c in coordinates ):
                coordinates.add((x, y))
            counter +=1
        assert len(coordinates) == m, f"len food coords is {len(coordinates)} but must be {m}"
        return list(coordinates)
    
    def _gen_player_pos(self,x_coord, y_coord, l, m):
        adjacent_coords = set()
        counter = 0
        while len(adjacent_coords)<m:
            if counter >= 10000:
                print(f"Warning: Max attempts (10000) reached in _gen_player_pos")
                break
            x, y = self.np_random.integers(x_coord - l, x_coord + l, endpoint=True), self.np_random.integers(y_coord - l, y_coord + l, endpoint=True)
            if (x,y) not in self.food_pos and x >=0 and y>=0 and x<=self.rows-1 and y<=self.cols-1:
                #if all(abs(x - cx) >= 1 or abs(y - cy) >= 1 for cx, cy in adjacent_coords):
                if self._dist_cell((x_coord, y_coord), (x,y)) <= min(self.level**2, 6):
                    adjacent_coords.add((x, y))
            counter +=1
        assert len(adjacent_coords) == m, f"len player coords is {len(adjacent_coords)} but must be {m}"
        return list(adjacent_coords)

    def manual_pos(self):
        if self.level <= 2:
            self.food_pos = self._gen_coord(2,6,self.max_num_food)
        else:
            self.food_pos = self._gen_coord(1,7,self.max_num_food)
        self.fix_target()
        self.player_pos = self._gen_player_pos(self.target_pos[0], self.target_pos[1], l=self.level, m=self.num_agents)

        self.spawn_players(self.min_player_level, self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_num_food,
            min_levels=self.min_food_level,
            max_levels=self.max_food_level
            if self.max_food_level is not None
            else np.array([sum(player_levels[:3])] * self.max_num_food),
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        if self.render_mode == "human":
            self.render()

        self.prev_dist = [100 for _ in range(self.num_agents)]
        self.field = np.zeros(self.field_size, np.int32)
        self.grid = np.zeros(self.field_size, np.int32)

        self.manual_pos()

        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()
        
        return self._get_obs(), {}

    def step(self, actions, count=True):
        if count:
            self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()
        loaded = False

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])
            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0
            self.grid[frow, fcol] = 0 #grid-obs
            self.food_pos.remove((frow,fcol))
            loaded = True

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        curr_dists = []
        for i, p in enumerate(self.players):
            p.score += p.reward

            if self.reward_potential:
                curr_dist = self._dist_cell(self.target_pos, p.position)
                curr_dists.append(curr_dist)
                if curr_dist > 1:
                    if self.prev_dist[i] <= curr_dist:
                        p.reward -= 0.1
                    else:
                        p.reward += 0.1
                self.prev_dist[i] = curr_dist

        if self.load_punish:
            if all(x == 1 for x in curr_dists):
                if not all(a == Action.LOAD for a in actions):
                    for p in self.players:
                        p.reward -= 0.5

        rewards = {i:p.reward for i,p in enumerate(self.players)}
        done = self._game_over
        truncated = False
        if loaded and not done: self.fix_target()
        
        return self._get_obs(), rewards, done, truncated, {}

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
