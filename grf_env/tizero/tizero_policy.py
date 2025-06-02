#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Official (and extended) implementation of the Tizero agent for the Google Research Football environment.
https://github.com/OpenRL-Lab/TiZero
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

from tizero.policy_network import PolicyNetwork
from openrl_utils import openrl_obs_deal, _t2n
from goal_keeper import agent_get_action

from tizero_critic import CriticNetwork

class OpenRLAgent():
    def __init__(self):
        rnn_shape = [1,1,1,512]
        self.rnn_hidden_state = [np.zeros(rnn_shape, dtype=np.float32) for _ in range (11)]
        self.actor = PolicyNetwork()
        self.actor.load_state_dict(torch.load( os.path.dirname(os.path.abspath(__file__)) + '/actor.pt', map_location=torch.device("cpu")))
        self.actor.eval()

        self.rnn_hidden_state_critic = np.zeros([1,1,512], dtype=np.float32)
        self.critic = CriticNetwork()
        self.critic.load_state_dict(torch.load( os.path.dirname(os.path.abspath(__file__)) + '/critic.pt', map_location=torch.device("cpu")))
        self.critic.eval()

    def get_action(self,raw_obs,idx, probs=False, all_acts=False, rnn=True):
        if idx == 0:
            re_action = [[0]*19]
            re_action_index = agent_get_action(raw_obs)[0]
            re_action[0][re_action_index] = 1
            return re_action, None, None if probs else re_action

        openrl_obs = openrl_obs_deal(raw_obs)

        obs = openrl_obs['obs']
        obs = np.concatenate(obs.reshape(1, 1, 330))
        if rnn:
            rnn_hidden_state = np.concatenate(self.rnn_hidden_state[idx])
        else:
            rnn_hidden_state = np.concatenate(np.zeros([1,1,512], dtype=np.float32))
        if all_acts:
            avail_actions = np.ones(20)
        else:
            avail_actions = np.zeros(20)
            avail_actions[:19] = openrl_obs['available_action']
        avail_actions = np.concatenate(avail_actions.reshape([1, 1, 20]))
        with torch.no_grad():
            actions, rnn_hidden_state, action_probs, dist_entropy = self.actor(obs, rnn_hidden_state, available_actions=avail_actions, deterministic=True)
        if actions[0][0] == 17 and raw_obs["sticky_actions"][8] == 1:
            actions[0][0] = 15
        self.rnn_hidden_state[idx] = np.array(np.split(_t2n(rnn_hidden_state), 1))

        re_action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        re_action[0][actions[0]] = 1

        return re_action, action_probs, dist_entropy if probs else re_action
         
    def get_val(self,raw_obs, rnn=True):
        openrl_obs = openrl_obs_deal(raw_obs)
        obs = openrl_obs['share_obs']
        obs = np.concatenate(obs.reshape(1, 1, 220))
        rnn_hidden_state = np.concatenate(self.rnn_hidden_state_critic)
        with torch.no_grad():
            value, rnn_hidden_state = self.critic(obs, rnn_hidden_state)
        if rnn:
            self.rnn_hidden_state_critic = np.array(np.split(_t2n(rnn_hidden_state), 1))
        return value
    
    def get_action_probs(self, obs, all_acts=False, rnn=True):
        idx = obs['controlled_player_index'] % 11 #same as "active" if 11 players
        return self.get_action(obs,idx, probs=True, all_acts=all_acts, rnn=rnn)
    
    def get_value(self, obs, scale=4.0, rnn=False):
        return float(self.get_val(obs, rnn)[0][0]) / scale

agent = OpenRLAgent()

def my_controller(obs_list, action_space_list, is_act_continuous=False):
    idx = obs_list['controlled_player_index'] % 11
    del obs_list['controlled_player_index'] #deletes the "key" in the dict
    action = agent.get_action(obs_list,idx)
    return action

def get_action_probs(obs, all_acts=False, rnn=True):
    idx = obs['controlled_player_index'] % 11 #same as "active" if 11 players
    return agent.get_action(obs,idx, probs=True, all_acts=all_acts, rnn=rnn)

def get_value(obs, rnn=False):
    return float(agent.get_value(obs, rnn))