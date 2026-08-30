import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GATConv
from torch.nn.modules.module import Module


class GAT(Module):
    """Multi-layer GAT backbone.

    Supports three view construction modes:
      - None:   mean-pooled single-view embedding [N, D]
      - 'head': attention heads of the last layer as views [N, H, D]
      - 'layer': intermediate layer outputs as views [N, L, D]
    """
    def __init__(self, args, activation=F.elu, negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        self.args = args
        self.activation = activation
        self.negative_slope = negative_slope
        self.residual = residual
        self.gat_layers = self.build_gnn_layers()

    def build_gnn_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(GATConv(self.args.input_dim, self.args.hidden_dim,
                              self.args.num_gnn_heads,
                              self.args.feat_drop_rate, self.args.attn_drop_rate,
                              self.negative_slope, self.residual, self.activation))
        for _ in range(1, self.args.num_gnn_layers - 1):
            layers.append(GATConv(self.args.num_gnn_heads * self.args.hidden_dim,
                                  self.args.hidden_dim, self.args.num_gnn_heads,
                                  self.args.feat_drop_rate, self.args.attn_drop_rate,
                                  self.negative_slope, self.residual, self.activation))
        layers.append(GATConv(self.args.num_gnn_heads * self.args.hidden_dim,
                              self.args.hidden_dim, self.args.num_gnn_heads,
                              self.args.feat_drop_rate, self.args.attn_drop_rate,
                              self.negative_slope, self.residual, self.activation))
        return layers

    def forward(self, inputs: torch.Tensor, g, view_type=None):
        """Args:
            inputs: [N, D_in]
            g: dgl graph (or list of blocks from NeighborSampler)
        """
        is_blocks = isinstance(g, list)

        if view_type == 'layer':
            h = inputs
            layer_outputs = []
            for l in range(self.args.num_gnn_layers - 1):
                gl = g[l] if is_blocks else g
                h = self.gat_layers[l](gl, h)          # [N, H, D]
                h_view = h.mean(1)                     # [N, D]
                layer_outputs.append(F.normalize(h_view, dim=1))
                h = h.flatten(1)                       # [N, H*D]
            gl = g[-1] if is_blocks else g
            final_h = self.gat_layers[-1](gl, h)       # [N, H, D]
            h_final = torch.mean(final_h, dim=1)       # [N, D]
            layer_outputs.append(F.normalize(h_final, dim=1))
            return torch.stack(layer_outputs, dim=1)   # [N, L, D]

        h = inputs
        for l in range(self.args.num_gnn_layers - 1):
            gl = g[l] if is_blocks else g
            h = self.gat_layers[l](gl, h).flatten(1)
        gl = g[-1] if is_blocks else g
        final_h = self.gat_layers[-1](gl, h)           # [N, H, D]

        if view_type == 'head':
            return F.normalize(final_h, dim=2)         # [N, H, D]
        return F.normalize(final_h.mean(dim=1), dim=1)  # [N, D]
