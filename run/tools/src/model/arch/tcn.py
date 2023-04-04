import torch
import torch.nn as nn
from ..builder import ARCHS
from torch.nn.utils import weight_norm
from einops.layers.torch import Rearrange
from .base import Base


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                (1, kernel_size),
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )
        self.net = nn.Sequential(
            self.pad,
            self.conv1,
            self.relu,
            self.dropout,
            self.pad,
            self.conv2,
            self.relu,
            self.dropout,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = in_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@ARCHS.register_module()
class TCN(Base):
    """
    Transformer with patch embedding layer
    """
    def __init__(self, num_layers=2, kernel_size=2, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.name = "TCN"
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        # build backbone
        self.backbone = nn.Sequential(Rearrange('b l c -> b c l'),
                                      TemporalConvNet(self.embedding_size, [self.hidden_size]*self.num_layers, kernel_size=self.kernel_size,
                                                      dropout=self.dropout),
                                      Rearrange('b c l -> b l c')
                                      )
