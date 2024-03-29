import torch
from torchvision.ops import MLP
import numpy as np
import torch.nn as nn
import itertools
from typing import Tuple, Dict, Sequence

ActT = np.ndarray | torch.Tensor
ObsT = Dict[str, np.ndarray]
Device = torch.device | str

class ObsPreprocessNet(nn.Module):
    """A simple CNN that generates an intermediate representation of the image observation"""

    def __init__(
        self,
        single_obs_shape: Sequence[int],
        intermed_rep_size:int,
        hidden_channels: Sequence[int]=[32, 64, 64],
        use_batchnorm:bool=True,
        use_dropout:bool=True,
        device:Device="cpu",
    ):
        super(ObsPreprocessNet, self).__init__()
        input_channel = (
            2 * single_obs_shape[0]
        )  # 2 images in the obs dict will be stacked
        image_shape = single_obs_shape[1:]
        channels = [input_channel] + list(hidden_channels)
        self.convs = nn.ModuleList(
            [
                self.conv_layer(
                    in_channels,
                    out_channels,
                    use_batchnorm=use_batchnorm,
                    use_dropout=use_dropout,
                )
                for in_channels, out_channels in itertools.pairwise(channels)
            ]
        )
        # the size of outputs is divided by 4 for each hidden layer (stride is 2)
        final_layer_size = np.prod(image_shape).item() // (4 ** len(hidden_channels))
        self.fc = nn.Linear(
            in_features=hidden_channels[-1] * final_layer_size,
            out_features=intermed_rep_size,
        )
        self.device = device

    def conv_layer(self, in_channels: int, out_channels: int, use_batchnorm: bool, use_dropout: bool) -> nn.Module:
        """Helper function for creating a convolutional layer with batch normalization, ReLU activation, and dropout"""
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                bias=not use_batchnorm,  # bias will be eliminated anyway by the following BatchNorm2d
                stride=2,
                padding=2,
            )
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(p=0.5))
        return nn.Sequential(*layers)

    def forward(self, obs: ObsT, state=None, info: Dict={}) -> Tuple[torch.Tensor, None]:
        # info is not needed here
        del info
        obs_list: list[torch.Tensor] = [
            torch.from_numpy(arr).float().to(dtype=torch.float32)
            for arr in obs.values()
        ]
        x = torch.cat(obs_list, 1).to(self.device)
        # Apply convolutional layers one by one
        for conv in self.convs:
            x = conv(x)
        # Flatten the output of the third convolutional layer
        x = x.view(x.size(0), -1)
        # Apply linear layer
        logits = self.fc(x)
        return logits, state


class ActPreprocessNet(nn.Module):
    """A warpper of MLP that generates an intermediate representation of the action"""

    def __init__(
        self,
        single_act_shape: Sequence[int],
        intermed_rep_size: int,
        hidden_features: Sequence[int]=[32, 64, 64],
        use_batchnorm:bool=True,
        use_dropout:bool=True,
        device:Device="cpu",
    ):
        super(ActPreprocessNet, self).__init__()
        input_features = int(np.prod(single_act_shape))
        norm = nn.BatchNorm1d if use_batchnorm else None
        dropout_rate = 0.5 if use_dropout else 0.0
        self.net = MLP(
            input_features,
            list(hidden_features) + [intermed_rep_size],
            norm_layer=norm,
            bias=not use_batchnorm,
            dropout=dropout_rate,
        )
        self.device = device

    def forward(self, act:ActT, state=None, info:Dict={}) -> Tuple[torch.Tensor, None]:
        # info is not needed
        del info
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        return self.net(act), state


class ActorNet(nn.Module):
    """
    Simple actor network. Will create an actor operated in continuous
    action space with structure of preprocess_net ---> action_shape.
    """

    def __init__(
        self,
        preprocess_net: ObsPreprocessNet,
        preprocess_net_output_dim: int,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int]=[16, 32],
        max_action: float = 1.0,
    ):
        super().__init__()
        self.preprocess = preprocess_net
        output_dim = int(np.prod(action_shape))
        input_dim = preprocess_net_output_dim
        self.final_mlp = MLP(
            input_dim, list(hidden_sizes) + [output_dim], nn.BatchNorm1d, bias=False
        )
        self._max = max_action

    def forward(
        self,
        obs:ObsT,
        state=None,
        info:Dict={},
    ) -> Tuple[torch.Tensor, None]:
        """Mapping: obs -> logits -> action."""
        # info is not needed
        del info
        logits, hidden = self.preprocess(obs, state)
        logits = self._max * torch.tanh(self.final_mlp(logits))
        return logits, hidden


class CriticNet(nn.Module):
    def __init__(
        self,
        act_preprocess_net: ActPreprocessNet,
        act_preprocess_net_output_dim: int,
        obs_preprocess_net: ObsPreprocessNet,
        obs_preprocess_net_output_dim: int,
        hidden_sizes:Sequence[int]=[16, 32],
        device:Device="cpu",
    ):
        super().__init__()
        self.device = device
        self.act_preprocess = act_preprocess_net
        self.obs_preprocess = obs_preprocess_net
        input_dim = act_preprocess_net_output_dim + obs_preprocess_net_output_dim
        output_dim = 1
        self.final_mlp = MLP(input_dim, list(hidden_sizes) + [output_dim])

    def forward(
        self,
        obs:ObsT,
        act:ActT,
        info:Dict={},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        # info is not needed
        del info
        s_intermed, _ = self.obs_preprocess(obs)
        a_intermed, _ = self.act_preprocess(act)
        logits = torch.cat([s_intermed, a_intermed], dim=1)
        logits = self.final_mlp(logits)
        return logits

