import torch
import torch.nn as nn
import torch.nn.functional as F


class N2PContrastiveLoss(nn.Module):
    """Node-to-Prototype contrastive loss.

    For node i (anchor from view 1), prototypes of its class in the same and
    cross views are positives; prototypes of all other classes are negatives.
    Intra-class prototype importance weights w(i,k,m) are computed via
    softmax over prototype similarities and sharpen over epochs through the
    annealed temperature tau_t = tau * gamma^epoch.
    """
    def __init__(self, tau=0.5, gamma=0.99, use_weight=True, tau_p=None,
                 single_anchor=False, w_same_view=False, separate_view=False):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.use_weight = use_weight
        self.tau_p = tau_p if tau_p is not None else tau
        # anchor from view 1 only
        self.single_anchor = single_anchor
        # w from same-view similarity (True) or cross-view (False)
        self.w_same_view = w_same_view
        # per-view denominator averaged (True) vs. joint denominator (False)
        self.separate_view = separate_view

    def forward(self, views, prototypes, prototype_labels, labels, epoch,
                prototype_mask=None):
        """
        Args:
            views: [V, N, D]
            prototypes: [V, C*K, D]
            labels: [N]
            prototype_mask: optional [C, K] active-prototype mask. Inactive
                prototypes are excluded from both the denominator and the
                positive weighting.
        """
        V, N, D = views.shape
        K = torch.sum(prototype_labels == prototype_labels[0]).item()
        C = len(torch.unique(prototype_labels))

        if self.separate_view:
            return self._forward_separate(
                views, prototypes, prototype_labels, labels, epoch,
                V, N, D, K, C, prototype_mask)

        views = F.normalize(views, dim=2)
        prototypes = F.normalize(prototypes, dim=2)

        tau_t = self.tau * (self.gamma ** epoch)

        if prototype_mask is not None:
            mask_4d = prototype_mask.unsqueeze(0).unsqueeze(0)  # [1,1,C,K]
        else:
            mask_4d = None

        total_loss = 0.0
        anchor_views = [0] if self.single_anchor else list(range(V))

        for anchor_view in anchor_views:
            h = views[anchor_view]  # [N, D]

            sim_all_views = []
            w_sim_all_views = []

            for proto_view in range(V):
                p = prototypes[proto_view]  # [C*K, D]
                sim = torch.matmul(h, p.t())
                sim_all_views.append(sim)

                if self.w_same_view:
                    h_same = views[proto_view]
                    w_sim_all_views.append(torch.matmul(h_same, p.t()))

            sim_all_views = torch.stack(sim_all_views, dim=1).view(N, V, C, K)

            if self.w_same_view:
                w_sim_all_views = torch.stack(w_sim_all_views, dim=1).view(N, V, C, K)

            # positives
            y_expand = labels.view(N, 1, 1, 1).expand(-1, V, 1, K)
            pos_sim = torch.gather(sim_all_views, dim=2, index=y_expand).squeeze(2)  # [N, V, K]

            if mask_4d is not None:
                pos_mask = torch.gather(mask_4d.expand(N, V, C, K), dim=2,
                                        index=y_expand).squeeze(2)  # [N, V, K]
            else:
                pos_mask = None

            if self.use_weight:
                if self.w_same_view:
                    pos_w_sim = torch.gather(w_sim_all_views, dim=2,
                                             index=y_expand).squeeze(2)
                    if pos_mask is not None:
                        pos_w_sim = pos_w_sim.masked_fill(pos_mask == 0, -1e9)
                    w = torch.softmax(pos_w_sim / self.tau_p, dim=-1)
                else:
                    if pos_mask is not None:
                        pos_sim_masked = pos_sim.masked_fill(pos_mask == 0, -1e9)
                    else:
                        pos_sim_masked = pos_sim
                    w = torch.softmax(pos_sim_masked / self.tau_p, dim=-1)

                pos_exp = torch.exp(pos_sim / tau_t) * w
                if pos_mask is not None:
                    pos_exp = pos_exp * pos_mask
            else:
                pos_exp = torch.exp(pos_sim / tau_t)
                if pos_mask is not None:
                    pos_exp = pos_exp * pos_mask

            numerator = pos_exp.sum(dim=(1, 2))

            # denominator
            all_exp = torch.exp(sim_all_views / tau_t)
            if mask_4d is not None:
                all_exp = all_exp * mask_4d

            pos_exp_unweighted = torch.exp(pos_sim / tau_t)
            if pos_mask is not None:
                pos_exp_unweighted = pos_exp_unweighted * pos_mask
            pos_exp_unweighted = pos_exp_unweighted.sum(dim=(1, 2))
            denominator = numerator + (all_exp.sum(dim=(1, 2, 3)) - pos_exp_unweighted)

            loss = -torch.log(numerator / (denominator + 1e-12))
            total_loss += loss.mean()

        total_loss /= len(anchor_views)
        return total_loss

    def _forward_separate(self, views, prototypes, prototype_labels, labels,
                          epoch, V, N, D, K, C, prototype_mask=None):
        """Per-view variant: denominator only contains prototypes of a single
        view, and the final loss is averaged over views."""
        views = F.normalize(views, dim=2)
        prototypes = F.normalize(prototypes, dim=2)

        tau_t = self.tau * (self.gamma ** epoch)

        total_loss = 0.0
        anchor_views = [0] if self.single_anchor else list(range(V))

        for anchor_view in anchor_views:
            h = views[anchor_view]

            for proto_view in range(V):
                p = prototypes[proto_view]
                sim = torch.matmul(h, p.t()).view(N, C, K)  # [N, C, K]

                y_expand = labels.view(N, 1, 1).expand(-1, 1, K)
                pos_sim = torch.gather(sim, dim=1, index=y_expand).squeeze(1)  # [N, K]

                if prototype_mask is not None:
                    pos_mask = torch.gather(
                        prototype_mask.unsqueeze(0).expand(N, C, K),
                        dim=1, index=y_expand).squeeze(1)
                else:
                    pos_mask = None

                if self.use_weight:
                    if pos_mask is not None:
                        pos_sim_masked = pos_sim.masked_fill(pos_mask == 0, -1e9)
                    else:
                        pos_sim_masked = pos_sim
                    w = torch.softmax(pos_sim_masked / self.tau_p, dim=-1)
                    pos_exp = torch.exp(pos_sim / tau_t) * w
                    if pos_mask is not None:
                        pos_exp = pos_exp * pos_mask
                else:
                    pos_exp = torch.exp(pos_sim / tau_t)
                    if pos_mask is not None:
                        pos_exp = pos_exp * pos_mask

                numerator = pos_exp.sum(dim=1)  # [N]

                all_exp = torch.exp(sim / tau_t)
                if prototype_mask is not None:
                    all_exp = all_exp * prototype_mask.unsqueeze(0)
                pos_exp_unweighted = torch.exp(pos_sim / tau_t)
                if pos_mask is not None:
                    pos_exp_unweighted = pos_exp_unweighted * pos_mask
                pos_exp_unweighted = pos_exp_unweighted.sum(dim=1)

                denominator = numerator + (all_exp.sum(dim=(1, 2)) - pos_exp_unweighted)
                loss = -torch.log(numerator / (denominator + 1e-12))
                total_loss += loss.mean()

        total_loss /= (len(anchor_views) * V)
        return total_loss
