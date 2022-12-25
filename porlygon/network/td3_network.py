import torch
import numpy as np
import torch.nn as nn
import itertools


class ConvNet(nn.Module):
    """A simple and stright-forward Convolutional Network"""

    def __init__(
        self,
        state_shape,
        action_shape,
        hidden_channels=[32, 64, 64],
        use_batchnorm=True,
        use_dropout=True,
    ):
        super(ConvNet, self).__init__()
        input_channel = state_shape[1]
        image_shape = state_shape[2:]
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
        # the size of outputs is divided by 4for each hidden layer (stride is 2)
        final_layer_size = np.prod(image_shape).item() // (4 ** len(hidden_channels))
        self.fc = nn.Linear(
            in_features=hidden_channels[-1] * final_layer_size,
            out_features=np.prod(action_shape).item(),
        )

    def conv_layer(self, in_channels, out_channels, use_batchnorm, use_dropout):
        """Helper function for creating a convolutional layer with batch normalization, ReLU activation, and dropout"""
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
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


if __name__ == "__main__":
    from torchinfo import summary

    n = ConvNet(state_shape=(6, 128, 128), action_shape=12)
    summary(n, (16, 6, 128, 128))
