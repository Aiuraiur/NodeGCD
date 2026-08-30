import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeManager(nn.Module):
    """Maintains prototype masks (active/inactive) and importance weights alpha.

    During progressive merging, merged-away prototypes are deactivated through
    `prototype_mask`; `alpha[c, k]` is the normalized contribution weight of
    prototype k to class c, estimated from node assignments.
    """
    def __init__(self, n_class, n_prototypes, hidden_dim, tau=0.5, device="cuda"):
        super().__init__()
        self.n_class = n_class
        self.n_prototypes = n_prototypes
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.device = device

        # active mask [C, K]
        self.register_buffer("prototype_mask", torch.ones(n_class, n_prototypes))
        # importance weights [C, K]
        self.register_buffer("alpha", torch.ones(n_class, n_prototypes) / n_prototypes)

    @torch.no_grad()
    def get_pseudo_labels(self, q):
        """q: [N, C] soft class assignment -> hard pseudo labels [N]."""
        return q.argmax(dim=-1)

    @torch.no_grad()
    def update_alpha(self, w, hard_count=False):
        """Update alpha from prototype assignment probabilities.

        Args:
            w: [N, V, C, K] prototype assignment probabilities.
            hard_count: if True, alpha from argmax counts; else from soft means.
        """
        if hard_count:
            proto_assign = w.argmax(dim=-1)                       # [N, V, C]
            one_hot = F.one_hot(proto_assign,
                                num_classes=self.n_prototypes).float()
            alpha = one_hot.sum(dim=(0, 1))                      # [C, K]
        else:
            alpha = w.mean(dim=(0, 1))                            # [C, K]

        alpha = alpha * self.prototype_mask
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-12)
        self.alpha.copy_(alpha)

    def get_num_active(self):
        return self.prototype_mask.sum(dim=1)
