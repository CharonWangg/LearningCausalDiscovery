import torch.nn as nn
from ..builder import ARCHS
from .base import Base
from .commons import DecoderHead
from ..utils import add_prefix

from einops.layers.torch import Rearrange


class FFNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=1, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(in_channels),
                                 nn.Linear(in_channels, out_channels*mlp_ratio, bias=False),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.LayerNorm(out_channels * mlp_ratio),
                                 nn.Linear(out_channels*mlp_ratio, out_channels, bias=False),
                                 )
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.ffn(x)
        x += res
        x = self.act(x)
        return x


@ARCHS.register_module()
class Autoencoder(Base):
    """
    Base architecture with patch embedding layer
    """
    def __init__(self, mlp_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.name = "Autoencoder"
        n_patch = self.input_length//self.patch_size
        # build embedding layer
        self.embedding = nn.Sequential(
                                        Rearrange(f"b l c -> b c l"),
                                        nn.Conv1d(
                                            self.in_channels,
                                            self.embedding_size,
                                            kernel_size=self.patch_size,
                                            stride=self.patch_size
                                        ),
                                        Rearrange(f"b c l -> b l c"),
        )

        # build backbone
        self.backbone = nn.Sequential(FFNBlock(self.embedding_size, self.hidden_size, mlp_ratio=mlp_ratio),
                                      nn.Flatten(),
                                      nn.Linear(n_patch*self.hidden_size*mlp_ratio,
                                                self.hidden_size))

        # build decode head
        self.head = DecoderHead(in_channels=self.hidden_size, num_classes=self.num_classes, losses=self.losses)
        self.head.model = nn.Sequential(nn.Linear(self.hidden_size, n_patch*self.hidden_size),
                                        nn.Unflatten(1, (n_patch, self.hidden_size)),
                                        FFNBlock(self.hidden_size, self.embedding_size, mlp_ratio=mlp_ratio),
                                        Rearrange(f"b l c -> b c l"),
                                        nn.ConvTranspose1d(self.embedding_size, self.in_channels,
                                                           kernel_size=self.patch_size, stride=self.patch_size),
                                        Rearrange(f"b c l -> b l c"),
                                        )

    def exact_feat(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        return x

    def forward_train(self, x, label):
        loss = dict()
        feat = self.exact_feat(x)

        decode_loss = self.head.forward_train(feat, x)
        loss.update(add_prefix(f"mainhead", decode_loss))

        # sum up all losses
        loss.update(
            {"loss": sum([loss[k] for k in loss.keys() if "loss" in k.lower()])}
        )

        # pack the output and losses
        return loss

    def forward_test(self, x, label=None):
        feat = self.exact_feat(x)
        res = self.head.forward_test(feat, x)

        # sum up all losses
        if label is not None:
            res.update(
                {"loss": sum([res[k] for k in res.keys() if "loss" in k.lower()])}
            )
        else:
            res.update({"loss": "Not available"})
        return res
