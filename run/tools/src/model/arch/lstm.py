import torch.nn as nn
from ..builder import ARCHS
from .base import Base, MeanPooler


@ARCHS.register_module()
class LSTM(Base):
    """
    LSTM with patch embedding layer
    """
    def __init__(self, losses, num_layers=2, bidirectional=True, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.name = "LSTM"
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # build backbone
        self.backbone = nn.LSTM(
            self.embedding_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # build head
        self.head = MeanPooler(in_channels=self.hidden_size*2, num_classes=self.num_classes, losses=losses)

    def exact_feat(self, x):
        x = self.embedding(x)
        x, (h, c) = self.backbone(x)
        return x