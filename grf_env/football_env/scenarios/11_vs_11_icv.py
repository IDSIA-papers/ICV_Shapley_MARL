# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os

import numpy as np

from . import *

def build_scenario(builder):
    #builder.config().control_all_players = True
    builder.config().game_duration = 3000 #env steps, per env step there are "physics_steps_per_frame"

    builder.config().left_team_difficulty = 1.0
    builder.config().right_team_difficulty = 1.0

    builder.config().deterministic = False

    #in beginning (init), first_team is right, then left
    if builder.EpisodeNumber() % 2 == 0:
        first_team = Team.e_Left
        second_team = Team.e_Right
    else:
        first_team = Team.e_Right
        second_team = Team.e_Left
    builder.SetTeam(first_team)
    
    # roles e.g. e_PlayerRole_GK = 0, e_PlayerRole_RM = 7, etc.
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=True) #0
    builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM) #7
    builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF) #9
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB) #2
    builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB) #1
    builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB) #1
    builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB) #3
    builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM) #5
    builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM) #5
    builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM) #5
    builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM) #6
    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=True)
    builder.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM)
    builder.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB)
    builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB)
    builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
    builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
    builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM)
    builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
    builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
    builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM)
