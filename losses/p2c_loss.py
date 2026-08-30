import torch
import torch.nn as nn
import torch.nn.functional as F


class P2CContrastiveLoss(nn.Module):
    """Prototype-to-Class contrastive loss.

    Each prototype is an anchor; the class-level prototype (aggregated over
    intra-class prototypes) of its own class is the positive, class-level
    prototypes of all other classes are negatives.
    """
    def __init__(self, tau=0.5, gamma=0.99, tau_p=None,
                 single_anchor=False, separate_view=False):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.tau_p = tau_p if tau_p is not None else tau
        self.single_anchor = single_anchor
        # per-view contrast (True) vs. joint denominator (False)
        self.separate_view = separate_view

    def forward(self, proto_reshape, class_proto, epoch=None, prototype_mask=None):
        """
        Args:
            proto_reshape: [V, C, K, D]
            class_proto: [V, C, D]
            prototype_mask: optional [C, K] active-prototype mask.
                Inactive prototypes are not anchors; classes without active
                prototypes are excluded from negatives.
        """
        V, C, K, D = proto_reshape.shape
        device = proto_reshape.device

        tau_t = self.tau * (self.gamma ** epoch) if epoch is not None else self.tau
        proto_reshape = F.normalize(proto_reshape, dim=-1)

        anchor_views = [0] if self.single_anchor else list(range(V))

        if prototype_mask is not None:
            active_mask = prototype_mask  # [C, K]
            active_classes = (active_mask.sum(dim=1) > 0)
        else:
            active_mask = None
            active_classes = torch.ones(C, dtype=torch.bool, device=device)

        if self.separate_view:
            total_loss, n_anchors = 0.0, 0
            for v in anchor_views:
                for c in range(C):
                    for k in range(K):
                        if active_mask is not None and active_mask[c, k] == 0:
                            continue
                        anchor = proto_reshape[v, c, k]

                        pos = F.normalize(class_proto[v:v+1, c, :], dim=-1)
                        neg_idx = [i for i in range(C)
                                   if i != c and active_classes[i]]
                        neg = F.normalize(class_proto[v:v+1, neg_idx, :], dim=-1)

                        sim_pos = torch.matmul(anchor, pos.T) / tau_t
                        sim_neg = torch.matmul(anchor, neg.reshape(-1, D).T) / tau_t

                        num = torch.exp(sim_pos).sum()
                        den = num + torch.exp(sim_neg).sum()
                        total_loss += -torch.log(num / den.clamp(min=1e-12))
                        n_anchors += 1
            if n_anchors == 0:
                return torch.tensor(0.0, device=device)
            return total_loss / n_anchors

        # joint denominator: prototypes of all views compete in one denominator
        total_loss, n_anchors = 0.0, 0
        for v in anchor_views:
            for c in range(C):
                for k in range(K):
                    if active_mask is not None and active_mask[c, k] == 0:
                        continue
                    anchor = proto_reshape[v, c, k]

                    pos = F.normalize(class_proto[:, c, :], dim=-1)      # [V, D]
                    neg_idx = [i for i in range(C)
                               if i != c and active_classes[i]]
                    neg = F.normalize(class_proto[:, neg_idx, :], dim=-1)

                    sim_pos = torch.matmul(anchor, pos.T) / tau_t
                    sim_neg = torch.matmul(anchor, neg.reshape(-1, D).T) / tau_t

                    num = torch.exp(sim_pos).sum()
                    den = num + torch.exp(sim_neg).sum()
                    total_loss += -torch.log(num / den.clamp(min=1e-12))
                    n_anchors += 1

        if n_anchors == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / n_anchors
