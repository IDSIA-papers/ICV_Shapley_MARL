from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
import torch.nn.functional as F
torch, nn = try_import_torch()

OBS_SHAPE = 12

class TagNetAgentSymmIndep(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self._inp_all = None

        self.inp1 = SlimFC(
            OBS_SHAPE,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        self.act_out = SlimFC(
            150,
            self.num_outputs,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

        self.inp_val = SlimFC(
            OBS_SHAPE,
            150,
            activation_fn= nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        self.val_out = SlimFC(
            150,
            1,
            activation_fn= None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._inp_all = input_dict["obs"]
        x = self.inp1(input_dict["obs"])
        x = self.act_out(x)
        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._inp_all is not None, "must call forward first!"
        x = self.inp_val(self._inp_all)
        x = self.val_out(x)
        return torch.reshape(x, [-1])

    def get_value(self, input_dict):
        self._inp_all = input_dict["obs"]
        return self.value_function()
