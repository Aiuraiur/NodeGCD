import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from .gat import GAT


class GNNModel(Module):
    """Graph backbone that returns L2-normalized node embeddings."""
    def __init__(self, args):
        super(GNNModel, self).__init__()
        if args.encoder_name == "gat":
            self.encoder = GAT(args)
        else:
            raise NotImplementedError(
                'encoder not supported: {}'.format(args.encoder_name))

    def forward(self, feats: torch.Tensor, adj, view_type=None):
        emb = self.encoder(feats, adj, view_type=view_type)
        if view_type is not None:
            return emb  # already normalized inside GAT
        return F.normalize(emb, dim=1)
