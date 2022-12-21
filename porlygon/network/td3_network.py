import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """A simple and stright-forward Convolutional Network"""

    def __init__(self, use_batchnorm=True, use_dropout=True):
        super(ConvNet, self).__init__()
        self.convs = nn.ModuleList(
            [
                self.conv_layer(
                    in_channels,
                    out_channels,
                    use_batchnorm=use_batchnorm,
                    use_dropout=use_dropout,
                )
                for in_channels, out_channels in [(6, 32), (32, 64), (64, 128)]
            ]
        )
        self.fc = nn.Linear(in_features=128 * 16 * 16, out_features=10)

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

    def forward(self, obs: dict, state=None, info={}):
        obs_list = [torch.from_numpy(arr).float() for arr in obs.values()]
        x = torch.cat(obs_list, 0)
        # Apply convolutional layers one by one
        for conv in self.convs:
            x = conv(x)
        # Flatten the output of the third convolutional layer
        x = x.view(x.size(0), -1)
        # Apply linear layer
        logits = self.fc(x)
        return logits, state
