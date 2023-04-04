import torch.nn as nn
from ..builder import ARCHS
from .base import Base


@ARCHS.register_module()
class Transformer(Base):
    """
    Transformer with patch embedding layer
    """
    def __init__(self, num_layers=4, nhead=8, **kwargs):
        super().__init__(**kwargs)
        self.name = "Transformer"

        # build backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=nhead,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)