import os
import torch
from tizero.policy_network import *
from tizero.openrl_utils import *

"""
Official (and extended) implementation of the Tizero critic for the Google Research Football environment.
https://github.com/OpenRL-Lab/TiZero
"""

class InputEncoder(nn.Module):
    def __init__(self):
        super(InputEncoder, self).__init__()
        fc_layer_num = 2
        fc_output_num = 64
        self.ball_owner_input_num = 35
        self.left_input_num = 88
        self.right_input_num = 88
        self.match_state_input_num = 9

        self.ball_owner_encoder = FcEncoder(fc_layer_num, self.ball_owner_input_num, fc_output_num)
        self.left_encoder = FcEncoder(fc_layer_num, self.left_input_num, fc_output_num)
        self.right_encoder = FcEncoder(fc_layer_num, self.right_input_num, fc_output_num)
        self.match_state_encoder = FcEncoder(fc_layer_num, self.match_state_input_num, self.match_state_input_num)

    def forward(self, x):
        ball_owner_vec = x[:, :self.ball_owner_input_num]
        left_vec = x[:, self.ball_owner_input_num : self.ball_owner_input_num + self.left_input_num]
        right_vec = x[:, self.ball_owner_input_num + self.left_input_num : self.ball_owner_input_num + self.left_input_num + self.right_input_num]
        match_state_vec = x[:, self.ball_owner_input_num + self.left_input_num + self.right_input_num:]

        ball_owner_output = self.ball_owner_encoder(ball_owner_vec)
        left_output = self.left_encoder(left_vec)
        right_output = self.right_encoder(right_vec)
        match_state_output = self.match_state_encoder(match_state_vec)

        return torch.cat([
            ball_owner_output,
            left_output,
            right_output,
            match_state_output
        ], 1)
    
class CriticNetwork(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(CriticNetwork, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.hidden_size = 256

        self.input_encoder = InputEncoder()
        self.base = FcEncoder(fc_num=3, input_size=201, output_size=256)
        self.rnn = RNNLayer(inputs_dim=self.hidden_size, outputs_dim=self.hidden_size,
                            recurrent_N=1,use_orthogonal=True,rnn_type='lstm')
        self.v_out = nn.Linear(in_features=256,out_features=1)
        self.to(device)

    def forward(self, obs, rnn_states, masks=np.concatenate(np.ones((1, 1, 1), dtype=np.float32))):
        obs = check(obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        obs = self.input_encoder(obs)
        obs = self.base(obs)
        obs, rnn_states = self.rnn(obs,rnn_states,masks)
        out = self.v_out(obs)
        return out, rnn_states

        
def get_critic():
    critic = CriticNetwork()
    critic.load_state_dict(torch.load( os.path.dirname(os.path.abspath(__file__)) + '/critic.pt', map_location=torch.device("cpu")))
    critic.eval()
    return critic

