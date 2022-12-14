import torch
from torchvision.ops import MLP
import numpy as np
import torch.nn as nn
import itertools


class ObsPreprocessNet(nn.Module):
    """A simple CNN that generates an intermediate representation of the image observation"""

    def __init__(
        self,
        single_obs_shape,
        intermed_rep_size,
        hidden_channels=[32, 64, 64],
        use_batchnorm=True,
        use_dropout=True,
    ):
        super(ObsPreprocessNet, self).__init__()
        input_channel = (
            2 * single_obs_shape[0]
        )  # 2 images in the obs dict will be stacked
        image_shape = single_obs_shape[1:]
        channels = [input_channel] + hidden_channels
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

    def conv_layer(self, in_channels, out_channels, use_batchnorm, use_dropout):
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

    def forward(self, obs, state=None, info={}):
        obs_list = [torch.from_numpy(arr).float() for arr in obs.values()]
        x = torch.cat(obs_list, 1)
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
        single_act_shape,
        intermed_rep_size,
        hidden_features=[32, 64, 64],
        use_batchnorm=True,
        use_dropout=True,
    ):
        super(ActPreprocessNet, self).__init__()
        input_features = single_act_shape[0]
        norm = nn.BatchNorm1d if use_batchnorm else None
        dropout_rate = 0.5 if use_dropout else 0.0
        self.net = MLP(
            input_features,
            hidden_features + [intermed_rep_size],
            norm_layer=norm,
            bias=not use_batchnorm,
            dropout=dropout_rate,
        )

    def forward(self, act, state=None, info={}):
        return self.net(act), state


class ActorNet(nn.Module):
    """Simple actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        preprocess_net_output_dim,
        action_shape,
        hidden_sizes=[],
        max_action: float = 1.0,
    ):
        super().__init__()
        self.preprocess = preprocess_net
        output_dim = int(np.prod(action_shape))
        input_dim = preprocess_net_output_dim
        self.last = MLP(
            input_dim, hidden_sizes + [output_dim], nn.BatchNorm1d, bias=False
        )
        self._max = max_action

    def forward(
        self,
        obs,
        state=None,
        info={},
    ):
        """Mapping: obs -> logits -> action."""
        logits, hidden = self.preprocess(obs, state)
        logits = self._max * torch.tanh(self.last(logits))
        return logits, hidden


class CriticNet(nn.Module):
    def __init__(
        self,
        act_preprocess_net: nn.Module,
        act_preprocess_net_output_dim,
        obs_preprocess_net: nn.Module,
        obs_preprocess_net_output_dim,
        device="cpu",
        hidden_sizes=[],
    ):
        super().__init__()
        self.device = (device,)
        self.act_preprocess = act_preprocess_net
        self.obs_preprocess = obs_preprocess_net
        input_dim = act_preprocess_net_output_dim + obs_preprocess_net_output_dim
        output_dim = 1
        self.last = MLP(input_dim, hidden_sizes + [output_dim])

    def forward(
        self,
        s,
        a,
        info={},
    ):
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)  # type: ignore
        a = torch.as_tensor(a, device=self.device, dtype=torch.float32)  # type: ignore
        s_intermed = self.obs_preprocess(s)
        a_intermed = self.act_preprocess(a)
        logits = torch.cat([s_intermed, a_intermed], dim=1)
        logits = self.last(logits)
        return logits


if __name__ == "__main__":
    from torchinfo import summary

    n = ObsPreprocessNet(single_obs_shape=(3, 128, 128), intermed_rep_size=12)
    summary(n, (16, 6, 128, 128))
