from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

OBS_SHAPE = 86

class LBFModel(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self._v1 = None

        self.inp1 = SlimFC(
            OBS_SHAPE,
            300,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        self.act_out = SlimFC(
            300,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

        self.inp1_val = SlimFC(
            3*(OBS_SHAPE+1),
            400,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.val_out = SlimFC(
            400,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._v1 = torch.cat((input_dict["obs"]["obs_1_own"], input_dict["obs"]["act_1_own"], input_dict["obs"]["obs_2"], input_dict["obs"]["act_2"], input_dict["obs"]["obs_3"], input_dict["obs"]["act_3"]),dim=1)

        x = input_dict["obs"]["obs_1_own"]
        x = self.inp1(x)
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._v1 is not None, "must call forward first!"
        x = self.inp1_val(self._v1)
        x = self.val_out(x)
        return torch.reshape(x, [-1])
    
    def get_value(self, input_dict):
        self._v1 = torch.cat((input_dict["obs_1_own"], input_dict["act_1_own"], input_dict["obs_2"], input_dict["act_2"], input_dict["obs_3"], input_dict["act_3"]),dim=1)
        return self.value_function()